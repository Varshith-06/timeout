<#
.SYNOPSIS
  Build the Processing-job image and push it to ECR.

.DESCRIPTION
  Deploys the timeout-precompute stack (ECR repo + job bucket + execution role),
  then builds the container from the repository root and pushes it.

  The build context is the repo root, not this directory, because the image needs
  src/, scripts/, models/phase2/ and yolo11l.pt. Expect ~1.5 GB and a slow first
  push; later pushes only send changed layers, which is usually just the app.

.EXAMPLE
  .\deploy\aws\sagemaker\build_and_push.ps1
#>
param(
    [string]$StackName = "timeout-precompute",
    [string]$Tag       = "latest",
    [string]$Region    = ""
)

$ErrorActionPreference = "Stop"

$awsCmd = Get-Command aws -ErrorAction SilentlyContinue
if ($awsCmd) {
    $aws = $awsCmd.Source
} elseif (Test-Path "C:\Program Files\Amazon\AWSCLIV2\aws.exe") {
    $aws = "C:\Program Files\Amazon\AWSCLIV2\aws.exe"
} else {
    throw "AWS CLI not found. Install it, or open a new shell if you just did."
}

docker info --format "{{.ServerVersion}}" | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "Docker is not running. Start Docker Desktop and wait for the whale icon to settle."
}

$repoRoot = (Resolve-Path "$PSScriptRoot\..\..\..").Path
$template = Join-Path $PSScriptRoot "cloudformation.yaml"

$regionArgs = @()
if ($Region) { $regionArgs = @("--region", $Region) }

Write-Host "deploying stack $StackName ..."
$deployOut = & $aws cloudformation deploy `
    --template-file $template `
    --stack-name $StackName `
    --capabilities CAPABILITY_NAMED_IAM `
    @regionArgs
if ($LASTEXITCODE -ne 0 -and "$deployOut" -notmatch "No changes to deploy") {
    Write-Host $deployOut
    throw "cloudformation deploy failed (exit $LASTEXITCODE)"
}

$outputsJson = & $aws cloudformation describe-stacks --stack-name $StackName `
    --query "Stacks[0].Outputs" --output json @regionArgs
if ($LASTEXITCODE -ne 0) { throw "could not read stack outputs" }
$outputs = @{}
foreach ($o in ($outputsJson | ConvertFrom-Json)) { $outputs[$o.OutputKey] = $o.OutputValue }

$repoUri  = $outputs["RepositoryUri"]
$registry = $repoUri.Split("/")[0]
$image    = "$($repoUri):$Tag"

if (-not $Region) {
    $Region = & $aws configure get region
    if (-not $Region) { throw "no region configured; run 'aws configure' or pass -Region" }
}

Write-Host "logging in to $registry ..."
$password = & $aws ecr get-login-password --region $Region
if ($LASTEXITCODE -ne 0) { throw "ecr get-login-password failed" }
$password | docker login --username AWS --password-stdin $registry
if ($LASTEXITCODE -ne 0) { throw "docker login failed" }

Write-Host "building $image (context: $repoRoot) ..."
docker build -f (Join-Path $PSScriptRoot "Dockerfile") -t $image $repoRoot
if ($LASTEXITCODE -ne 0) { throw "docker build failed" }

Write-Host "pushing (first push uploads ~1.5 GB) ..."
docker push $image
if ($LASTEXITCODE -ne 0) { throw "docker push failed" }

Write-Host ""
Write-Host "pushed: $image"
Write-Host "Next: python deploy\aws\sagemaker\run_job.py --video <clip.mp4> --calib <calib.json>"
