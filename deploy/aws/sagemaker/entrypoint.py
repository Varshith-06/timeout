"""SageMaker Processing entrypoint for the web-app pre-compute pass.

Processing jobs follow a directory contract rather than an argv one: declared
inputs are downloaded into the container before it starts, and whatever lands in
the declared output directory is uploaded to S3 after it exits. This shim maps
those directories onto ``scripts/build_webapp.py``'s CLI so the pipeline itself
needs no changes to run locally *or* on SageMaker.

    /opt/ml/processing/input/video/*.mp4    the clip
    /opt/ml/processing/input/calib/*.json   calibration(s), and optional roster
    /opt/ml/processing/output/              clip.mp4 + recommendations.json + index.html

Any extra flags passed as container arguments are forwarded verbatim to
build_webapp, so stride/cadence/etc. are tunable per job without a rebuild.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

IN = Path("/opt/ml/processing/input")
OUT = Path("/opt/ml/processing/output")


def _fail(msg: str) -> int:
    # Anything on stderr lands in the job's CloudWatch log stream, which is the
    # only place to debug a Processing job after the container is gone.
    print(f"entrypoint: {msg}", file=sys.stderr)
    return 2


def main(argv=None) -> int:
    passthrough = list(sys.argv[1:] if argv is None else argv)

    videos = sorted(p for p in (IN / "video").glob("*") if p.suffix.lower() in {".mp4", ".mov", ".mkv"})
    if not videos:
        return _fail(f"no video found under {IN / 'video'}")
    if len(videos) > 1:
        print(f"entrypoint: {len(videos)} videos present, using {videos[0].name}")
    video = videos[0]

    jsons = sorted((IN / "calib").glob("*.json"))
    if not jsons:
        return _fail(f"no calibration json found under {IN / 'calib'}")

    # A roster file is optional and lives in the same input channel; it is
    # distinguished by name rather than by schema-sniffing.
    rosters = [p for p in jsons if "roster" in p.name.lower()]
    calibs = [p for p in jsons if p not in rosters]
    if not calibs:
        return _fail("input/calib contained only a roster; need at least one calibration json")

    OUT.mkdir(parents=True, exist_ok=True)
    args = ["--video", str(video), "--out", str(OUT)]

    # Multiple calibrations = multiple camera shots merged into one timeline;
    # each must carry its own click-time (calibrate.py --time).
    if len(calibs) > 1:
        args += ["--shots", *[str(p) for p in sorted(calibs)]]
    else:
        args += ["--calib", str(calibs[0])]
    if rosters:
        args += ["--roster", str(rosters[0])]

    # Jersey OCR needs easyocr, which is not in the image (it pulls a second
    # torch model set and downloads weights at runtime — bad for a hermetic
    # job). Callers can still pass --no-jersey explicitly; this just makes the
    # default match what the container can actually do.
    if "--no-jersey" not in passthrough:
        args.append("--no-jersey")
    args += passthrough

    print(f"entrypoint: build_webapp {' '.join(args)}", flush=True)
    from scripts.build_webapp import main as build_webapp

    rc = build_webapp(args)
    if rc == 0:
        produced = sorted(p.name for p in OUT.iterdir())
        print(f"entrypoint: wrote {produced} -> uploading to S3")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
