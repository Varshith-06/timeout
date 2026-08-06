"""Train the court-keypoint model from existing click calibrations (roadmap 4.6).

    python scripts/train_keypoints.py \
        --pair data/video/clips/youtube/QIiLBgFQmOs.f136.mp4 shot1.json shot2.json shot4.json \
        --pair data/video/clips/youtube/game2_J6RT3BHHvPU.mp4 g2shot1.json g2shot3.json \
        --epochs 40

Harvests labelled frames from each (video, calibrations) pair, trains the
heatmap network, and writes ``models/court_keypoints.pt``.

**The validation split holds out whole camera shots, not random frames.** Frames
from one shot are near-duplicates of each other, so a random split would report a
pixel error that says nothing about the only question that matters: does this
work on a shot nobody calibrated? Held-out-shot validation answers that, and it
is why the reported numbers are worse — and honest.

Two error figures are reported per epoch:
  * **px** — mean distance between predicted and labelled keypoints, in source
    pixels, over visible points only.
  * **ft** — median reprojection error of the homography solved *from the
    predictions*, in court feet. This is the number the pipeline actually cares
    about; ``solve_homography`` rejects a frame above 2 ft.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.perception.calibrate import Calibration           # noqa: E402
from src.perception.homography import solve_homography      # noqa: E402
from src.perception.keypoint_data import (                  # noqa: E402
    export_images, harvest_calibration, jitter_colour, load_dataset, warp_sample,
    write_manifest)
from src.perception.keypoint_model import (                 # noqa: E402
    DEFAULT_WEIGHTS, IN_H, IN_W, K, build_net, decode_heatmaps, gaussian_targets,
    masked_mse, preprocess, save_checkpoint)
from src.perception.keypoints import KEYPOINT_XY            # noqa: E402
from src.perception.video import VideoSource                # noqa: E402


def harvest(pairs, stride, max_span, out_dir):
    """Run the auto-labeller over every (video, calibrations) pair.

    Each shot is flushed to disk as soon as it is harvested, so peak memory is
    one shot's frames rather than the whole dataset's.
    """
    records = []
    for spec in pairs:
        video_path, calib_paths = spec[0], spec[1:]
        video = VideoSource(video_path)
        print(f"{Path(video_path).name}: {video.frame_count} frames @ {video.fps:.1f} fps")
        for cp in calib_paths:
            calib = Calibration.load(cp)
            calib._name = Path(cp).stem          # groups train/val by shot
            got = harvest_calibration(video, calib, stride=stride, max_span_sec=max_span)
            records.extend(export_images(got, out_dir, offset=len(records)))
            print(f"  {Path(cp).name:18s} t={calib.time:>7} -> {len(got):4d} frames")
            del got                              # release this shot's images
        video.release()
    if not records:
        raise SystemExit("no frames harvested — check the calibration --time values")
    print(f"harvested {len(records)} labelled frames -> {out_dir}")
    write_manifest(records, out_dir)
    return load_dataset(out_dir)


def shot_of(record) -> str:
    """Which calibration produced this record (everything before the '@')."""
    return record["source"].split("@")[0]


def make_datasets(manifest, root, val_shots, seed):
    import cv2
    import torch

    class KeypointDataset(torch.utils.data.Dataset):
        def __init__(self, records, augment: bool):
            self.records = records
            self.augment = augment
            self.root = Path(root) / "images"

        def __len__(self):
            return len(self.records)

        def __getitem__(self, i):
            r = self.records[i]
            bgr = cv2.imread(str(self.root / r["image"]))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            kp = np.array(r["kp_px"], dtype=float)
            vis = np.array(r["visible"], dtype=bool)

            # Downscale to the network's input size *before* augmenting. The warp
            # is the expensive step and preprocess() would resize to exactly this
            # anyway, so doing it at source resolution costs ~6x for no gain.
            h0, w0 = rgb.shape[:2]
            rgb = cv2.resize(rgb, (IN_W, IN_H), interpolation=cv2.INTER_AREA)
            kp = kp * np.array([IN_W / w0, IN_H / h0])

            if self.augment:
                rng = np.random.default_rng(seed * 100003 + i * 7919 + random.randrange(1 << 30))
                rgb, kp, vis = warp_sample(rgb, kp, vis, rng)
                rgb = jitter_colour(rgb, rng)

            hm, mask = gaussian_targets(kp, vis, (IN_W, IN_H))
            return (torch.from_numpy(preprocess(rgb)),
                    torch.from_numpy(hm), torch.from_numpy(mask))

    records = manifest["records"]
    val = [r for r in records if shot_of(r) in val_shots]
    train = [r for r in records if shot_of(r) not in val_shots]
    return KeypointDataset(train, True), KeypointDataset(val, False)


MIN_INLIERS = 6      # see evaluate(): 4 points fit any homography exactly


def evaluate(net, loader, device, records_iter):
    """Held-out keypoint pixel error, reprojection error (ft), and solve rate.

    ``px`` is the primary number and the one checkpoints are selected on: it is
    the direct, ungameable measure of whether the model finds the landmarks.

    ``ft`` is reported only over frames whose homography rests on at least
    ``MIN_INLIERS`` correspondences. A homography through 4 points has zero
    residual by construction, so without that floor a model that finds four
    landmarks and hallucinates the rest scores a *better* reprojection error than
    one that finds twenty — and selecting on it would actively prefer the
    degenerate model.
    """
    import torch
    net.eval()
    px_errors, ft_errors, n_frames, n_solved, n_used = [], [], 0, 0, []
    with torch.no_grad():
        for (x, hm, mask), recs in zip(loader, records_iter):
            pred = net(x.to(device)).float().cpu().numpy()
            for p, r in zip(pred, recs):
                kp_true = np.array(r["kp_px"], dtype=float)
                vis = np.array(r["visible"], dtype=bool)
                # Labels are stored at source resolution; decode to the same frame.
                pts, conf = decode_heatmaps(p, _record_size(r))
                # Score only the points that actually reach the solver: visible in
                # the frame AND confident. Averaging over every labelled landmark
                # instead would be dominated by the ones the model correctly
                # declines to place, which the pipeline never uses.
                scored = vis & (conf >= 0.5)
                if scored.any():
                    px_errors.extend(
                        np.linalg.norm(pts[scored] - kp_true[scored], axis=1).tolist())
                n_frames += 1
                n_used.append(int(scored.sum()))
                keep = conf >= 0.5                      # matches solve_homography's gate
                if keep.sum() >= MIN_INLIERS:
                    res = solve_homography(pts[keep], KEYPOINT_XY[keep], max_reproj_ft=1e9)
                    if np.isfinite(res.median_reproj_ft) and res.n_points >= MIN_INLIERS:
                        ft_errors.append(res.median_reproj_ft)
                        if res.median_reproj_ft <= 2.0:
                            n_solved += 1
    net.train()
    return (float(np.median(px_errors)) if px_errors else float("nan"),
            float(np.median(ft_errors)) if ft_errors else float("nan"),
            n_solved / max(n_frames, 1),
            float(np.mean(n_used)) if n_used else 0.0)


_SIZE_CACHE: dict = {}


def _record_size(r):
    """Source image size for a record; labels are in these coordinates."""
    return _SIZE_CACHE.get(r["image"], (1280, 720))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", nargs="+", action="append", required=False, metavar="VIDEO CALIB",
                    help="a video followed by its calibration json(s); repeatable")
    ap.add_argument("--dataset", default="out/keypoints_ds",
                    help="where harvested frames live (reused if --skip-harvest)")
    ap.add_argument("--skip-harvest", action="store_true",
                    help="reuse an already-harvested dataset directory")
    ap.add_argument("--val-shots", nargs="*", default=None,
                    help="calibration stems to hold out (default: one automatically)")
    ap.add_argument("--stride", type=int, default=5, help="frame stride when harvesting")
    ap.add_argument("--max-span", type=float, default=30.0,
                    help="seconds to propagate either side of each anchor")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--workers", type=int, default=0, help=">0 needs spawn-safe Windows")
    ap.add_argument("--out", default=DEFAULT_WEIGHTS)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import cv2
    import torch

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.skip_harvest:
        manifest = load_dataset(args.dataset)
        print(f"reusing {len(manifest['records'])} frames from {args.dataset}")
    else:
        if not args.pair:
            raise SystemExit("--pair is required unless --skip-harvest")
        manifest = harvest(args.pair, args.stride, args.max_span, args.dataset)

    # Cache each image's true size so decoded predictions land in label space.
    root = Path(args.dataset) / "images"
    for r in manifest["records"]:
        img = cv2.imread(str(root / r["image"]))
        _SIZE_CACHE[r["image"]] = (img.shape[1], img.shape[0])

    shots = sorted({shot_of(r) for r in manifest["records"]})
    if args.val_shots is None:
        # Hold out the shot with the median frame count — not the biggest (too
        # much training data lost) nor the smallest (too noisy a metric).
        by_size = sorted(shots, key=lambda s: sum(shot_of(r) == s for r in manifest["records"]))
        val_shots = {by_size[len(by_size) // 2]}
    else:
        val_shots = set(args.val_shots)
    print(f"shots: {shots}\nheld out for validation: {sorted(val_shots)}")

    train_ds, val_ds = make_datasets(manifest, args.dataset, val_shots, args.seed)
    if len(val_ds) == 0:
        raise SystemExit(f"validation split is empty; --val-shots must be among {shots}")
    print(f"train {len(train_ds)} frames · val {len(val_ds)} frames")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    net = build_net().to(device)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"model: {n_params/1e6:.2f}M params, {K} keypoints")

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        drop_last=len(train_ds) > args.batch_size)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    val_batches = [val_ds.records[i:i + args.batch_size]
                   for i in range(0, len(val_ds.records), args.batch_size)]

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=max(1, args.epochs * len(train_loader)))

    # Checkpoint selection follows the product goal: the fraction of held-out
    # frames that yield a usable homography, tie-broken by how tight it is. Both
    # halves matter — solve-rate alone could be won by a model that is loosely
    # right everywhere, reprojection alone by one that is precise on four points
    # and blind elsewhere.
    best_score, best = (-1.0, float("inf")), {}
    for epoch in range(1, args.epochs + 1):
        total, nb = 0.0, 0
        for x, hm, mask in train_loader:
            x, hm, mask = x.to(device), hm.to(device), mask.to(device)
            opt.zero_grad(set_to_none=True)
            loss = masked_mse(net(x), hm, mask)
            loss.backward()
            opt.step()
            sched.step()
            total += float(loss.item()); nb += 1

        px, ft, rate, used = evaluate(net, val_loader, device, val_batches)
        score = (rate, -(ft if np.isfinite(ft) else 1e9))
        flag = ""
        if score > best_score:
            best_score = score
            best = {"val_px": px, "val_reproj_ft": ft, "val_solve_rate": rate,
                    "val_points_used": used, "epoch": epoch}
            save_checkpoint(net, args.out, meta={
                "val_shots": sorted(val_shots), "train_frames": len(train_ds),
                "val_frames": len(val_ds), **best})
            flag = "  <- saved"
        print(f"epoch {epoch:3d}/{args.epochs}  loss {total/max(nb,1):.5f}  "
              f"val {px:6.1f} px  {ft:5.2f} ft  solved {100*rate:3.0f}%  "
              f"kp {used:4.1f}{flag}")

    print(f"\nbest held-out epoch {best.get('epoch')}: "
          f"{best.get('val_px', float('nan')):.1f} px median on used points, "
          f"{best.get('val_reproj_ft', float('nan')):.2f} ft, "
          f"{100*best.get('val_solve_rate', 0):.0f}% of frames solved within the 2 ft gate "
          f"({best.get('val_points_used', 0):.1f} keypoints used per frame)")
    print(f"weights -> {args.out}")
    if best.get("val_solve_rate", 0) < 0.5:
        print("NOTE: the model solves under half of held-out frames, so most will fall back "
              "to the propagated calibration. It is trained on a handful of calibrated "
              "shots — calibrating more shots is the fix, not more epochs.")


if __name__ == "__main__":
    main()
