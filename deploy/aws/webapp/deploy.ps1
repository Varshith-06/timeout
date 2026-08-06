<#
.SYNOPSIS
  Deploy the pre-built web app to S3 + CloudFront.

.DESCRIPTION
  Creates (or updates) the timeout-webapp stack, uploads the three artifacts the
  page needs with correct content types and cache headers, and invalidates the
  CloudFront cache so a redeploy is visible immediately.

  The first run takes ~15 minutes because CloudFront has to propagate the new
  distribution; later runs take seconds.

.EXAMPLE
  .\deploy\aws\webapp\deploy.ps1
  .\deploy\aws\webapp\deploy.ps1 -Source out\webapp_sagemaker
  .\deploy\aws\webapp\deploy.ps1 -NoCloudFront   # new account, CloudFront not yet available
#>
param(
    [string]$Source    = "out\webapp",
    [string]$StackName = "timeout-webapp",
    [string]$Region    = "",
    # A new AWS account cannot create CloudFront distributions until Support
    # verifies it. This falls back to a plain S3 website bucket: public and
    # HTTP-only, but live today. Re-run without the switch once verified.
    [switch]$NoCloudFront
)

$ErrorActionPreference = "Stop"

# The installer does not update the PATH of already-running shells, so fall back
# to the default install location rather than failing on a fresh install.
$awsCmd = Get-Command aws -ErrorAction SilentlyContinue
if ($awsCmd) {
    $aws = $awsCmd.Source
} elseif (Test-Path "C:\Program Files\Amazon\AWSCLIV2\aws.exe") {
    $aws = "C:\Program Files\Amazon\AWSCLIV2\aws.exe"
} else {
    throw "AWS CLI not found. Install it, or open a new shell if you just did."
}

$repoRoot = (Resolve-Path "$PSScriptRoot\..\..\..").Path
$template = if ($NoCloudFront) {
    Join-Path $PSScriptRoot "cloudformation-s3website.yaml"
} else {
    Join-Path $PSScriptRoot "cloudformation.yaml"
}
$srcDir   = if ([System.IO.Path]::IsPathRooted($Source)) { $Source } else { Join-Path $repoRoot $Source }

if (-not (Test-Path $srcDir)) {
    throw "no such directory: $srcDir  (run scripts\build_webapp.py first)"
}
# Only these three are served; anything else in out\webapp is a debug artifact.
$assets = @(
    @{ Name = "index.html";            Type = "text/html; charset=utf-8"; Cache = "no-cache" },
    @{ Name = "recommendations.json";  Type = "application/json";         Cache = "no-cache" },
    @{ Name = "clip.mp4";              Type = "video/mp4";                Cache = "public, max-age=31536000, immutable" }
)
foreach ($a in $assets) {
    if (-not (Test-Path (Join-Path $srcDir $a.Name))) {
        throw "missing $($a.Name) in $srcDir  (run scripts\build_webapp.py first)"
    }
}

$regionArgs = @()
if ($Region) { $regionArgs = @("--region", $Region) }

Write-Host "deploying stack $StackName ..."
$deployOut = & $aws cloudformation deploy `
    --template-file $template `
    --stack-name $StackName `
    --capabilities CAPABILITY_IAM `
    @regionArgs
# 'No changes to deploy' is reported as a failure exit code; it is a success here.
if ($LASTEXITCODE -ne 0 -and "$deployOut" -notmatch "No changes to deploy") {
    Write-Host $deployOut
    throw "cloudformation deploy failed (exit $LASTEXITCODE)"
}

$outputsJson = & $aws cloudformation describe-stacks --stack-name $StackName `
    --query "Stacks[0].Outputs" --output json @regionArgs
if ($LASTEXITCODE -ne 0) { throw "could not read stack outputs" }
$outputs = @{}
foreach ($o in ($outputsJson | ConvertFrom-Json)) { $outputs[$o.OutputKey] = $o.OutputValue }

$bucket = $outputs["BucketName"]
$distId = $outputs["DistributionId"]
$siteUrl = $outputs["SiteUrl"]
Write-Host "bucket: $bucket"

foreach ($a in $assets) {
    $path = Join-Path $srcDir $a.Name
    $mb = [math]::Round((Get-Item $path).Length / 1MB, 1)
    Write-Host "  upload $($a.Name) ($mb MB)"
    & $aws s3 cp $path "s3://$bucket/$($a.Name)" `
        --content-type $a.Type `
        --cache-control $a.Cache `
        --only-show-errors @regionArgs
    if ($LASTEXITCODE -ne 0) { throw "upload of $($a.Name) failed" }
}

# No distribution in the S3-website fallback, so there is no edge cache to bust —
# the objects are already authoritative the moment the upload finishes.
if ($distId) {
    Write-Host "invalidating CloudFront cache ..."
    & $aws cloudfront create-invalidation --distribution-id $distId --paths "/*" `
        --query "Invalidation.Id" --output text @regionArgs | Out-Null
    if ($LASTEXITCODE -ne 0) { Write-Warning "invalidation failed; new files may take up to 24h to appear" }
}

Write-Host ""
Write-Host "deployed: $siteUrl"
if ($distId) {
    Write-Host "(a brand-new distribution needs ~15 min before it answers; redeploys are immediate)"
} else {
    Write-Host "(S3 website fallback: live immediately, but HTTP-only and un-cached."
    Write-Host " re-run without -NoCloudFront once the account is verified for CloudFront)"
}
