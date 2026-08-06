# AWS deployment

Two independent pieces:

| | What | Cost while up |
|---|---|---|
| `webapp/` | The demo itself — S3 + CloudFront static hosting | ~$0.50/month at portfolio traffic |
| `sagemaker/` | The pre-compute pass as a SageMaker Processing job | ~$0.15 per run, $0 idle |

The web app is meant to stay up. The Processing job is a batch pipeline: it runs,
writes its output to S3, and the instance is torn down automatically — there is
no endpoint left running and nothing to switch off afterwards.

## Prerequisites

```powershell
aws configure          # access key, secret, and a region (us-east-1 is a good default)
pip install boto3      # only needed for run_job.py
```

The IAM user needs permissions for CloudFormation, S3 and CloudFront (plus ECR,
SageMaker and IAM for part 2). A freshly created user has none, and every call
fails with `AccessDenied` — attach a policy in the console before starting, since
a user cannot grant itself permissions. Verify with:

```powershell
aws sts get-caller-identity
aws cloudformation list-stacks --stack-status-filter CREATE_COMPLETE
```

Docker Desktop must be running for `build_and_push.ps1`.

## 1. Deploy the web app

```powershell
python scripts\build_webapp.py --video <clip.mp4> --shots shot1.json shot2.json shot4.json
.\deploy\aws\webapp\deploy.ps1
```

`deploy.ps1` creates the stack, uploads `index.html`, `recommendations.json` and
`clip.mp4` with correct content types, invalidates the cache, and prints the URL.
A brand-new CloudFront distribution needs ~15 minutes to propagate before it
answers; redeploys are immediate.

**The bucket is private.** CloudFront reads it through an Origin Access Control,
so the objects are not directly reachable over S3 URLs.

### If the account is not yet verified for CloudFront

A new AWS account cannot create CloudFront distributions until Support enables
it, and the stack fails on the `Distribution` resource with:

```
Your account must be verified before you can add new CloudFront resources.
To verify your account, please contact AWS Support. (Status Code: 403)
```

That is an account state, not a template problem. Open a case (free, under
**Account and billing** → **CloudFront**) quoting the request ID from the error,
and in the meantime deploy without a CDN:

```powershell
.\deploy\aws\webapp\deploy.ps1 -Source out\webapp_heatnets_fixed -NoCloudFront
```

`-NoCloudFront` swaps in `cloudformation-s3website.yaml`: the same bucket served
as an S3 website, live in about a minute. The trade-offs are real and worth
stating — the bucket is **world-readable** and the endpoint is **HTTP-only with
no edge cache**. S3 website endpoints do serve Range requests, so the `<video>`
element still scrubs.

Re-run *without* the switch once the account is verified. Both templates use the
same `SiteBucket` logical ID and bucket name, so CloudFormation upgrades the
existing stack in place: the bucket goes back to private, the distribution and
OAC are added, and the URL becomes HTTPS. There is nothing to tear down first.

A failed `Distribution` create leaves the stack in `ROLLBACK_COMPLETE`, and the
bucket survives it by `DeletionPolicy: Retain`. Both must go before you retry, or
the next create collides on the bucket name:

```powershell
aws cloudformation delete-stack --stack-name timeout-webapp
aws cloudformation wait stack-delete-complete --stack-name timeout-webapp
aws s3 rb s3://timeout-webapp-<account-id>
```

**The chatbot degrades gracefully.** A static origin has no `POST /chat`, so the
page's fetch fails and it falls back to the built-in rule-based assistant, which
needs no API key. That is the intended behaviour for the public demo — the Claude
backend stays a local-only feature rather than a hosted key you would have to pay
for and rotate.

## 2. Run the pre-compute pass on SageMaker

```powershell
.\deploy\aws\sagemaker\build_and_push.ps1
python deploy\aws\sagemaker\run_job.py `
    --video data\video\clips\youtube\QIiLBgFQmOs.f136.mp4 `
    --calib shot1.json shot2.json shot4.json `
    --roster roster_game2.json
```

### Building the image without a local Docker daemon

`build_and_push.ps1` needs a working Docker daemon. On Windows Home that means
the WSL2 backend — Hyper-V is not available on that edition — and WSL2 can fail
in ways no Docker-side fix reaches (a broken Host Compute Service will refuse to
create any VM, so `wsl -d <distro> -e echo ok` fails for *every* distro, not just
Docker's). The container path is then unavailable locally while the pipeline it
builds remains perfectly runnable in the cloud.

`build_image_codebuild.py` builds it in CodeBuild instead:

```powershell
python deploy\aws\sagemaker\build_image_codebuild.py
```

It packs the build context (only the paths the Dockerfile COPYs — sending the
repo would mean uploading tens of GB of clips), uploads it to the job bucket,
runs the CodeBuild project from the stack, and streams the build log. **The
Dockerfile is unchanged**; only the machine running `docker build` moves. It is
also faster than building locally, because the layers reach ECR from inside AWS
rather than over a home uplink.

> **New accounts start with a CodeBuild quota of zero.** `StartBuild` then fails
> with `AccountLimitExceededException: Cannot have more than 0 builds in queue`.
> That is an account limit, not a broken project. Request it — the quota is
> adjustable and the increase is free:
>
> ```powershell
> aws service-quotas request-service-quota-increase `
>     --service-code codebuild --quota-code L-9D07B6EF --desired-value 1
> ```
>
> `L-9D07B6EF` is *Concurrently running builds for Linux/Small*, which is what
> `BUILD_GENERAL1_SMALL` consumes. Check progress with
> `aws service-quotas list-requested-service-quota-change-history --service-code codebuild`.

`run_job.py` stages the inputs to S3, starts the job, tails its CloudWatch logs,
and downloads the result to `out\webapp_sagemaker\` — which `deploy.ps1 -Source
out\webapp_sagemaker` will then publish.

### How it maps onto the Processing contract

SageMaker Processing is directory-driven, not argv-driven: declared inputs are
downloaded into the container before it starts, and whatever lands in the declared
output directory is uploaded to S3 when it exits. `entrypoint.py` adapts those
paths onto `build_webapp.py`'s existing CLI, so the same pipeline runs unchanged
locally and on SageMaker.

```
/opt/ml/processing/input/video/*.mp4    <- s3://<bucket>/jobs/<job>/input/video/
/opt/ml/processing/input/calib/*.json   <- s3://<bucket>/jobs/<job>/input/calib/
/opt/ml/processing/output/              -> s3://<bucket>/jobs/<job>/output/
```

### Choices worth knowing

- **CPU, not GPU.** The image installs CPU-only torch. The CUDA wheels add ~2.5 GB,
  and on a home connection the push dominates the wall-clock cost of the whole
  exercise. `ml.m5.2xlarge` finishes a clip in minutes.
- **Processing job, not an inference endpoint.** The pipeline is a batch pass that
  runs once per clip and writes JSON; there is no per-request inference in the
  product. A real-time endpoint would idle at GPU prices for a workload that never
  makes an online call.
- **Jersey OCR is off.** easyocr pulls a second model set and downloads weights at
  runtime, which would make the job non-hermetic. Pass `--` then `--no-jersey`'s
  absence only if you add easyocr to the image.
- **Least-privilege role.** The execution role is scoped to this project's bucket
  and ECR repository rather than using the managed `AmazonSageMakerFullAccess`
  policy, which grants account-wide S3 and SageMaker access.
- **`MaxRuntimeInSeconds` is a spend guard.** Default 1 hour; SageMaker stops the
  job at that point whether or not it finished.

## Teardown

The Processing job leaves nothing running. To remove the rest:

```powershell
# Web app — empty the bucket first; the stack retains it by design.
aws s3 rm s3://timeout-webapp-<account-id> --recursive
aws cloudformation delete-stack --stack-name timeout-webapp

# Pre-compute — delete images and job data, then the stack.
aws s3 rm s3://timeout-precompute-<account-id> --recursive
aws ecr delete-repository --repository-name timeout-precompute --force
aws cloudformation delete-stack --stack-name timeout-precompute
```

Both buckets use `DeletionPolicy: Retain`, so deleting a stack never silently
destroys uploaded content — you delete the bucket deliberately, as above.
