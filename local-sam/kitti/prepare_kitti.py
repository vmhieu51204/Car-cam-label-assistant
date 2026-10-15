"""Package one KITTI tracking sequence into a local-sam payload zip (CPU only; run on any machine).

    python kitti/prepare_kitti.py \
        --images ~/data/data_tracking_image_2 --labels ~/Downloads/data_tracking_label_2 \
        --sequence 0014 --out ~/kitti_payloads
    python kitti/prepare_kitti.py ... --sequence 0002 0004 --profile kitti5     # 5 classes -> kitti_0002_kitti5.zip
    python kitti/prepare_kitti.py ... --sequence 0014 --mots ~/BuildPhases/TrackEval/data/gt/kitti/kitti_mots_train
                                                     # + KITTI MOTS masks (car/pedestrian) for mask scoring

Zip layout (read by ``samlocal.payload`` like any repo payload; extra files are ignored by the model):

    frames/frame_000000.jpg ...            KITTI image_02 PNG -> JPEG (SAM3's folder loader is JPEG-based)
    frames.json                            source_frame_index = KITTI frame number
    benchmark-config.json                  text_prompts = profile defaults (kitti2: car, pedestrian;
                                           kitti5: car, van, truck, pedestrian, cyclist)
    kitti_manifest.json                    sequence, frame range, prompt -> class map, label checksum
    ground_truth/label_02/<seq>.txt        KITTI labels; used only by kitti/eval_kitti.py, never by the model
    ground_truth/instances_txt/<seq>.txt   KITTI MOTS masks (with --mots); used only by kitti/mots_kitti.py

Only the ``training`` split has labels. One zip = one sequence (a window must never span two sequences).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))  # kitti_metrics, also under python -I

from kitti_metrics import DEFAULT_PROMPTS, PROFILES, PROMPTS  # noqa: E402


def _find_dir(root: Path, kind: str, sequence: str) -> Path:
    """Accept the dataset root, ``training/`` or ``training/<kind>/`` for images (dir) or labels (file)."""
    root = root.expanduser()
    leaf = sequence if kind == "image_02" else f"{sequence}.txt"
    for candidate in (root / "training" / kind / leaf, root / kind / leaf, root / leaf):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"cannot find training/{kind}/{leaf} under {root}")


def build(images: Path, labels: Path, sequence: str, out: Path, *, start: int = 0, count: int | None = None,
          prompts: dict[str, str] | None = None, quality: int = 95, profile: str = "kitti2",
          text_prompts: list[str] | None = None, mots: Path | None = None) -> Path:
    """``prompts`` maps every accepted prompt to a class; ``text_prompts`` (default run set) is a subset of it."""
    prompts = prompts or PROMPTS[profile]
    text_prompts = text_prompts or [p for p in DEFAULT_PROMPTS[profile] if p in prompts] or list(prompts)
    if set(text_prompts) - set(prompts):
        raise ValueError(f"text prompts without a class: {sorted(set(text_prompts) - set(prompts))}")
    image_dir = _find_dir(images, "image_02", sequence)
    label_file = _find_dir(labels, "label_02", sequence)
    pngs = sorted(image_dir.glob("*.png"), key=lambda p: int(p.stem))
    if [int(p.stem) for p in pngs] != list(range(len(pngs))):
        raise ValueError(f"{image_dir}: frame numbers are not contiguous from 000000")
    stop = len(pngs) if count is None else min(len(pngs), start + count)
    if not 0 <= start < stop:
        raise ValueError(f"empty frame range [{start}, {stop}) for {len(pngs)} frames")
    label_bytes = label_file.read_bytes()
    mots_bytes = _find_dir(mots, "label_02", sequence).read_bytes() if mots else None
    gt_counts: Counter = Counter()
    gt_tracks: dict[str, set[str]] = {}
    for line in label_bytes.decode().splitlines():
        fields = line.split()
        if fields and start <= int(fields[0]) < stop:
            gt_counts[fields[2]] += 1
            gt_tracks.setdefault(fields[2], set()).add(fields[1])

    suffix = ("" if (start, stop) == (0, len(pngs)) else f"_{start:06d}-{stop - 1:06d}") + (
        "" if profile == "kitti2" else f"_{profile}")
    out.mkdir(parents=True, exist_ok=True)
    target = out / f"kitti_{sequence}{suffix}.zip"
    frames = []
    size = None
    with zipfile.ZipFile(target, "w", zipfile.ZIP_STORED) as archive:
        for position, png in enumerate(pngs[start:stop]):
            with Image.open(png) as image:
                size = image.size
                stream = io.BytesIO()
                image.convert("RGB").save(stream, format="JPEG", quality=quality)
            encoded = stream.getvalue()
            name = f"frames/frame_{position:06d}.jpg"
            archive.writestr(name, encoded)
            frames.append({"source_frame_index": int(png.stem), "relative_path": name, "sequence": sequence,
                           "sha256": hashlib.sha256(encoded).hexdigest(), "width": size[0], "height": size[1]})
        config = {"schema": "autolabel-sam3-benchmark-config/v1",
                  "configurations": [{"id": f"kitti-{sequence}", "mode": "text_prompt", "seed_type": "none",
                                      "text_prompts": text_prompts}]}
        manifest = {
            "schema": "local-sam-kitti/v1", "dataset": "KITTI tracking", "split": "training",
            "sequence": sequence, "camera": "image_02", "source_fps": 10,
            "frame_range": [start, stop], "num_frames": stop - start, "image_size": list(size or ()),
            "jpeg_quality": quality, "profile": profile, "classes": sorted({c for c, _ in PROFILES[profile].values()}),
            "default_prompts": text_prompts, "prompt_to_class": prompts,
            "labels": f"ground_truth/label_02/{sequence}.txt",
            "labels_sha256": hashlib.sha256(label_bytes).hexdigest(),
            "gt_rows_in_range": dict(gt_counts),
            "gt_tracks_in_range": {k: len(v) for k, v in gt_tracks.items()},
            "ground_truth_used_by_model": False,
        }
        if mots_bytes is not None:
            manifest["mots_labels"] = f"ground_truth/instances_txt/{sequence}.txt"
            manifest["mots_labels_sha256"] = hashlib.sha256(mots_bytes).hexdigest()
            archive.writestr(manifest["mots_labels"], mots_bytes)
        archive.writestr("frames.json", json.dumps({"frames": frames}))
        archive.writestr("benchmark-config.json", json.dumps(config, indent=2))
        archive.writestr("kitti_manifest.json", json.dumps(manifest, indent=2))
        archive.writestr(f"ground_truth/label_02/{sequence}.txt", label_bytes)
    print(json.dumps({"zip": str(target), "size_mib": round(target.stat().st_size / 2**20, 1),
                      **{k: manifest[k] for k in ("sequence", "frame_range", "num_frames", "image_size",
                                                  "gt_rows_in_range", "gt_tracks_in_range")}}, indent=2))
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", type=Path, required=True, help="data_tracking_image_2 (or its training/ folder)")
    parser.add_argument("--labels", type=Path, required=True, help="data_tracking_label_2 (or its training/ folder)")
    parser.add_argument("--sequence", nargs="+", required=True, help="one or more 4-digit sequences, e.g. 0014")
    parser.add_argument("--out", type=Path, required=True, help="output folder for kitti_<seq>.zip")
    parser.add_argument("--start", type=int, default=0, help="first KITTI frame (default 0)")
    parser.add_argument("--count", type=int, default=None, help="number of frames (default: whole sequence)")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="kitti2",
                        help="kitti2: car/person (official KITTI); kitti5: car/van/truck/pedestrian/cyclist")
    parser.add_argument("--prompts", default=None,
                        help="override prompt=class pairs, e.g. 'car=car,lorry=truck' (classes of the profile)")
    parser.add_argument("--text-prompts", default=None,
                        help="comma-separated prompts the benchmark runs by default (default: the profile's set)")
    parser.add_argument("--quality", type=int, default=95, help="JPEG quality")
    parser.add_argument("--mots", type=Path, default=None,
                        help="KITTI MOTS instances_txt folder (<seq>.txt, or TrackEval's kitti_mots_train with "
                             "label_02/); adds mask ground truth for kitti/mots_kitti.py")
    args = parser.parse_args(argv)
    prompts = None
    if args.prompts:
        prompts = dict(pair.split("=", 1) for pair in args.prompts.split(","))
        classes = {c for c, _ in PROFILES[args.profile].values()}
        if not set(prompts.values()) <= classes:
            parser.error(f"classes must be one of {sorted(classes)} for profile {args.profile}")
    for sequence in args.sequence:
        build(args.images, args.labels, f"{int(sequence):04d}", args.out, start=args.start, count=args.count,
              prompts=prompts, quality=args.quality, profile=args.profile,
              text_prompts=[p.strip() for p in args.text_prompts.split(",")] if args.text_prompts else None,
              mots=args.mots)
    return 0


if __name__ == "__main__":
    sys.exit(main())
