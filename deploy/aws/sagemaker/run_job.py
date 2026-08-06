"""Launch the pre-compute pass as a SageMaker Processing job and wait for it.

Uploads the clip and calibration files to the job bucket, starts the job against
the image in ECR, tails its CloudWatch logs, and downloads the resulting web-app
directory when it finishes.

    python deploy/aws/sagemaker/run_job.py \
        --video data/video/clips/youtube/QIiLBgFQmOs.f136.mp4 \
        --calib shot1.json shot2.json shot4.json \
        --roster roster_game2.json

Requires the `timeout-precompute` stack (cloudformation.yaml) and an image
pushed by build_and_push.ps1. Stack outputs supply the bucket, role, and image
URI, so nothing here is hardcoded to an account.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:  # pragma: no cover - environment guidance
    print("boto3 is required:  pip install boto3", file=sys.stderr)
    raise SystemExit(2)

STACK = "timeout-precompute"
CONTAINER_INPUT = "/opt/ml/processing/input"
CONTAINER_OUTPUT = "/opt/ml/processing/output"


def stack_outputs(cfn, stack: str) -> dict[str, str]:
    try:
        described = cfn.describe_stacks(StackName=stack)["Stacks"][0]
    except ClientError as e:
        raise SystemExit(
            f"stack '{stack}' not found ({e.response['Error']['Code']}). Deploy it first:\n"
            f"  aws cloudformation deploy --template-file deploy/aws/sagemaker/cloudformation.yaml "
            f"--stack-name {stack} --capabilities CAPABILITY_NAMED_IAM"
        )
    return {o["OutputKey"]: o["OutputValue"] for o in described.get("Outputs", [])}


def upload(s3, bucket: str, local: Path, key: str) -> str:
    size_mb = local.stat().st_size / 1e6
    print(f"  upload {local.name} ({size_mb:.1f} MB) -> s3://{bucket}/{key}")
    s3.upload_file(str(local), bucket, key)
    return f"s3://{bucket}/{key}"


def tail_logs(logs, job_name: str, stop_after: float) -> None:
    """Stream the job's log group until the job leaves the running state.

    Processing-job logs appear under /aws/sagemaker/ProcessingJobs with the job
    name as the stream prefix. The group does not exist until the container
    actually starts, so a missing group early on is expected, not an error.
    """
    group, token, seen_stream = "/aws/sagemaker/ProcessingJobs", None, None
    while time.time() < stop_after:
        try:
            if seen_stream is None:
                streams = logs.describe_log_streams(
                    logGroupName=group, logStreamNamePrefix=job_name, limit=1
                ).get("logStreams", [])
                if not streams:
                    time.sleep(5)
                    continue
                seen_stream = streams[0]["logStreamName"]
            kwargs = {"logGroupName": group, "logStreamName": seen_stream, "startFromHead": True}
            if token:
                kwargs["nextToken"] = token
            resp = logs.get_log_events(**kwargs)
            for event in resp["events"]:
                print(f"    {event['message'].rstrip()}")
            if resp["nextForwardToken"] == token:
                return
            token = resp["nextForwardToken"]
        except logs.exceptions.ResourceNotFoundException:
            time.sleep(5)
        except ClientError:
            return


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run build_webapp as a SageMaker Processing job")
    ap.add_argument("--video", required=True, help="local clip to process")
    ap.add_argument("--calib", nargs="+", required=True, help="calibration json(s), one per camera shot")
    ap.add_argument("--roster", help="optional roster json (filename must contain 'roster')")
    ap.add_argument("--stack", default=STACK)
    ap.add_argument("--region", default=None, help="defaults to the CLI's configured region")
    ap.add_argument("--instance-type", default="ml.m5.2xlarge",
                    help="CPU instance; the image ships CPU-only torch")
    ap.add_argument("--volume-size", type=int, default=30, help="GB of attached EBS")
    ap.add_argument("--max-runtime", type=int, default=3600,
                    help="seconds before SageMaker stops the job (a spend guard)")
    ap.add_argument("--download", default="out/webapp_sagemaker",
                    help="local dir for the job's output; '' to skip")
    ap.add_argument("--no-wait", action="store_true", help="submit and exit")
    ap.add_argument("extra", nargs="*",
                    help="extra build_webapp flags, after --  (e.g. -- --stride 8 --cadence 0.5)")
    args = ap.parse_args(argv)

    session = boto3.Session(region_name=args.region)
    region = session.region_name
    if not region:
        raise SystemExit("no AWS region configured; run 'aws configure' or pass --region")
    cfn, s3 = session.client("cloudformation"), session.client("s3")
    sm, logs = session.client("sagemaker"), session.client("logs")

    out = stack_outputs(cfn, args.stack)
    bucket, role, image = out["DataBucket"], out["ExecutionRoleArn"], out["RepositoryUri"] + ":latest"

    video = Path(args.video)
    calibs = [Path(c) for c in args.calib]
    roster = Path(args.roster) if args.roster else None
    for f in [video, *calibs, *( [roster] if roster else [] )]:
        if not f.exists():
            raise SystemExit(f"missing file: {f}")
    if roster and "roster" not in roster.name.lower():
        raise SystemExit(f"--roster filename must contain 'roster' so the container can "
                         f"tell it apart from a calibration: {roster.name}")

    job = f"timeout-precompute-{int(time.time())}"
    prefix = f"jobs/{job}"
    print(f"staging inputs to s3://{bucket}/{prefix}/")
    upload(s3, bucket, video, f"{prefix}/input/video/{video.name}")
    for c in calibs:
        upload(s3, bucket, c, f"{prefix}/input/calib/{c.name}")
    if roster:
        upload(s3, bucket, roster, f"{prefix}/input/calib/{roster.name}")

    print(f"starting {job} on {args.instance_type}")
    sm.create_processing_job(
        ProcessingJobName=job,
        RoleArn=role,
        AppSpecification={
            "ImageUri": image,
            "ContainerArguments": list(args.extra),
        },
        ProcessingResources={
            "ClusterConfig": {
                "InstanceCount": 1,
                "InstanceType": args.instance_type,
                "VolumeSizeInGB": args.volume_size,
            }
        },
        StoppingCondition={"MaxRuntimeInSeconds": args.max_runtime},
        ProcessingInputs=[
            {
                "InputName": "video",
                "S3Input": {
                    "S3Uri": f"s3://{bucket}/{prefix}/input/video",
                    "LocalPath": f"{CONTAINER_INPUT}/video",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                    "S3DataDistributionType": "FullyReplicated",
                },
            },
            {
                "InputName": "calib",
                "S3Input": {
                    "S3Uri": f"s3://{bucket}/{prefix}/input/calib",
                    "LocalPath": f"{CONTAINER_INPUT}/calib",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                    "S3DataDistributionType": "FullyReplicated",
                },
            },
        ],
        ProcessingOutputConfig={
            "Outputs": [
                {
                    "OutputName": "webapp",
                    "S3Output": {
                        "S3Uri": f"s3://{bucket}/{prefix}/output",
                        "LocalPath": CONTAINER_OUTPUT,
                        "S3UploadMode": "EndOfJob",
                    },
                }
            ]
        },
    )
    console = (f"https://{region}.console.aws.amazon.com/sagemaker/home"
               f"?region={region}#/processing-jobs/{job}")
    print(f"  console: {console}")
    if args.no_wait:
        return 0

    deadline = time.time() + args.max_runtime + 600
    tail_logs(logs, job, deadline)
    while True:
        desc = sm.describe_processing_job(ProcessingJobName=job)
        status = desc["ProcessingJobStatus"]
        if status in ("Completed", "Failed", "Stopped"):
            break
        time.sleep(15)
    print(f"job {status}")
    if status != "Completed":
        print(desc.get("FailureReason", "(no failure reason reported)"), file=sys.stderr)
        print(f"full logs: {console}", file=sys.stderr)
        return 1

    if args.download:
        dest = Path(args.download)
        dest.mkdir(parents=True, exist_ok=True)
        listed = s3.list_objects_v2(Bucket=bucket, Prefix=f"{prefix}/output/").get("Contents", [])
        for obj in listed:
            name = obj["Key"].rsplit("/", 1)[-1]
            if not name:
                continue
            print(f"  download {name} ({obj['Size'] / 1e6:.1f} MB)")
            s3.download_file(bucket, obj["Key"], str(dest / name))
        print(f"output -> {dest}\nNext: deploy/aws/webapp/deploy.ps1 -Source {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
