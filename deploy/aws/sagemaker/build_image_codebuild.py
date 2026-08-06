"""Build and push the job image using CodeBuild — no local Docker required.

    python deploy/aws/sagemaker/build_image_codebuild.py

Packs the build context, uploads it to the job bucket, runs the CodeBuild
project from the stack, and streams the build log. The Dockerfile is used
unchanged; only the machine doing the `docker build` moves from this laptop to
AWS.

This exists because Docker Desktop on Windows Home requires a working WSL2
backend, and WSL2 can fail in ways no Docker-side fix reaches — the whole
container path is then unavailable locally, while the pipeline it builds is
perfectly runnable in the cloud. It is also simply faster: the layers are pushed
to ECR from inside AWS rather than over a home uplink.

Use build_and_push.ps1 instead when a local Docker daemon does work; the two
produce the same image and the same ECR tag.
"""
from __future__ import annotations

import argparse
import sys
import time
import zipfile
from pathlib import Path

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:  # pragma: no cover - environment guidance
    print("boto3 is required:  pip install boto3", file=sys.stderr)
    raise SystemExit(2)

STACK = "timeout-precompute"

# Exactly what the Dockerfile COPYs, plus the Dockerfile itself. Sending the
# whole repo would mean uploading ~35 GB of clips and build output; this mirrors
# the re-includes in .dockerignore.
CONTEXT_PATHS = [
    "src",
    "scripts",
    "models/phase2",
    "yolo11l.pt",
    "deploy/aws/sagemaker/Dockerfile",
    "deploy/aws/sagemaker/entrypoint.py",
]
SKIP_SUFFIXES = (".pyc", ".pyo")
SKIP_DIRS = {"__pycache__", ".pytest_cache", ".git"}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


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


def pack_context(root: Path, dest: Path) -> int:
    """Zip the build context; returns the number of files written."""
    n = 0
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
        for rel in CONTEXT_PATHS:
            p = root / rel
            if not p.exists():
                raise SystemExit(f"missing from the build context: {rel}")
            if p.is_file():
                z.write(p, rel)
                n += 1
                continue
            for f in sorted(p.rglob("*")):
                if not f.is_file():
                    continue
                if f.suffix in SKIP_SUFFIXES or SKIP_DIRS & set(f.parts):
                    continue
                z.write(f, str(f.relative_to(root)).replace("\\", "/"))
                n += 1
    return n


def stream_log(logs, group: str, stream: str, token: str | None):
    """Print any new lines; returns the next token."""
    kwargs = {"logGroupName": group, "logStreamName": stream, "startFromHead": True}
    if token:
        kwargs["nextToken"] = token
    try:
        resp = logs.get_log_events(**kwargs)
    except (logs.exceptions.ResourceNotFoundException, ClientError):
        return token
    for event in resp["events"]:
        print(f"    {event['message'].rstrip()}")
    return resp["nextForwardToken"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build the job image in CodeBuild")
    ap.add_argument("--stack", default=STACK)
    ap.add_argument("--region", default=None, help="defaults to the CLI's configured region")
    ap.add_argument("--tag", default="latest", help="ECR image tag to push")
    ap.add_argument("--no-wait", action="store_true", help="start the build and exit")
    args = ap.parse_args(argv)

    session = boto3.Session(region_name=args.region)
    region = session.region_name
    if not region:
        raise SystemExit("no AWS region configured; run 'aws configure' or pass --region")
    cfn, s3 = session.client("cloudformation"), session.client("s3")
    cb, logs = session.client("codebuild"), session.client("logs")

    out = stack_outputs(cfn, args.stack)
    bucket, project = out["DataBucket"], out["ImageBuildProject"]
    repo_uri = out["RepositoryUri"]
    root = repo_root()

    archive = Path(f"{root}/.build-context.zip")
    print(f"packing build context from {root}")
    count = pack_context(root, archive)
    size_mb = archive.stat().st_size / 1e6
    print(f"  {count} files, {size_mb:.1f} MB")

    key = f"build/{int(time.time())}/source.zip"
    print(f"uploading -> s3://{bucket}/{key}")
    s3.upload_file(str(archive), bucket, key)
    archive.unlink(missing_ok=True)

    build = cb.start_build(
        projectName=project,
        sourceLocationOverride=f"{bucket}/{key}",
        environmentVariablesOverride=[
            {"name": "IMAGE_TAG", "value": args.tag, "type": "PLAINTEXT"},
        ],
    )["build"]
    build_id = build["id"]
    console = (f"https://{region}.console.aws.amazon.com/codesuite/codebuild/projects/"
               f"{project}/build/{build_id.replace(':', '%3A')}")
    print(f"started {build_id}\n  console: {console}")
    if args.no_wait:
        return 0

    group = stream = token = None
    status = "IN_PROGRESS"
    while status == "IN_PROGRESS":
        time.sleep(8)
        desc = cb.batch_get_builds(ids=[build_id])["builds"][0]
        status = desc["buildStatus"]
        info = desc.get("logs", {})
        group, stream = info.get("groupName", group), info.get("streamName", stream)
        if group and stream:
            token = stream_log(logs, group, stream, token)

    if group and stream:                       # drain whatever landed at the end
        stream_log(logs, group, stream, token)

    print(f"build {status}")
    if status != "SUCCEEDED":
        print(f"full log: {console}", file=sys.stderr)
        return 1
    print(f"image -> {repo_uri}:{args.tag}\nNext: python deploy/aws/sagemaker/run_job.py --video ... --calib ...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
