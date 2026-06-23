"""
Build a V5 middle-only paired dataset using the oblique Blender renderer.

This script is intentionally conservative: it reuses the existing v4 labels and
real targets, but replaces the old top-down/icon-like synthetic A image with a
new oblique 3D render from chess-set.blend.

For each selected middle-view sample:
  - render v5 RGB/seg/depth raw maps through Blender
  - perspective-crop them to 512x512
  - write paired [v5_rgb | real_target] image
  - write *_seg.png and *_depth.png control maps
"""
import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image


def count_pieces(fen):
    return sum(ch.isalpha() for ch in fen.split()[0])


def select_jobs(labels, split, limit):
    jobs = []
    for name, meta in sorted(labels.items()):
        if meta.get("synthetic_view") != "middle":
            continue
        if split != "all" and meta.get("split") != split:
            continue
        jobs.append({
            "name": name,
            "split": meta["split"],
            "fen": meta["fen"],
            "viewpoint": meta["viewpoint"],
            "game": meta["game"],
            "frame": meta["frame"],
            "pieces": count_pieces(meta["fen"]),
        })
        if limit and len(jobs) >= limit:
            break
    return jobs


def run(cmd, quiet=False):
    print("[run]", " ".join(str(x) for x in cmd))
    if quiet:
        subprocess.run(
            [str(x) for x in cmd],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
    else:
        subprocess.run([str(x) for x in cmd], check=True)


def real_target_from_v4(v4_dataset, job):
    packed = v4_dataset / job["split"] / f"{job['name']}.png"
    img = Image.open(packed).convert("RGB")
    w = img.width // 2
    return img.crop((w, 0, img.width, img.height))


def build_pair(rgb_path, real_img, out_path):
    rgb = Image.open(rgb_path).convert("RGB").resize((512, 512), Image.Resampling.BICUBIC)
    real = real_img.resize((512, 512), Image.Resampling.BICUBIC)
    pair = Image.new("RGB", (1024, 512))
    pair.paste(rgb, (0, 0))
    pair.paste(real, (512, 0))
    pair.save(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-dir", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--split", choices=["train", "test", "all"], default="train")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--blender", default="blender")
    ap.add_argument("--blend", default=None)
    ap.add_argument("--resolution", type=int, default=1000)
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--camera-y", type=float, default=1.05)
    ap.add_argument("--camera-z", type=float, default=0.90)
    ap.add_argument("--lens", type=float, default=48.0)
    ap.add_argument("--game-elevs", type=str, default="",
                    help="per-game camera elevation in deg, e.g. '2:44,4:40,5:38,6:42,7:42'. "
                         "camera-z is derived as camera_y*tan(elev) per sample's game.")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--quiet", action="store_true", help="suppress Blender/crop subprocess logs")
    args = ap.parse_args()

    game_elev = {}
    if args.game_elevs:
        for kv in args.game_elevs.split(","):
            g, e = kv.split(":")
            game_elev[int(g)] = float(e)

    project = Path(args.project_dir)
    v4_dataset = project / "datasets" / "chess_existing_geom_v4"
    out_dir = Path(args.out_dir) if args.out_dir else project / "datasets" / "chess_v5_oblique"
    raw_dir = out_dir / "_raw"
    crop_dir = out_dir / "_cropped"
    render_script = project / "v5_pipeline" / "render_oblique_blender.py"
    crop_script = project / "v5_pipeline" / "pil_perspective_crop.py"
    blend = Path(args.blend) if args.blend else project / "chess-set.blend"

    labels = json.loads((v4_dataset / "labels.json").read_text())
    jobs = select_jobs(labels, args.split, args.limit)
    if not jobs:
        raise RuntimeError("No jobs selected")

    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)
    for split in ["train", "test"]:
        (out_dir / split).mkdir(exist_ok=True)

    labels_out_path = out_dir / "labels.json"
    existing = {}
    if labels_out_path.exists() and not args.force:
        existing = json.loads(labels_out_path.read_text())

    out_labels = {}
    for idx, job in enumerate(jobs, 1):
        print(f"[{idx}/{len(jobs)}] {job['split']} {job['name']} pieces={job['pieces']}")
        job_raw = raw_dir / job["name"]
        job_crop = crop_dir / job["name"]
        meta = job_raw / f"{job['name']}_metadata.json"
        cropped_meta = job_crop / f"{job['name']}_cropped_metadata.json"
        pair_out = out_dir / job["split"] / f"{job['name']}.png"
        seg_out = out_dir / job["split"] / f"{job['name']}_seg.png"
        depth_out = out_dir / job["split"] / f"{job['name']}_depth.png"

        cam_z = args.camera_z
        if job["game"] in game_elev:
            cam_z = round(args.camera_y * math.tan(math.radians(game_elev[job["game"]])), 4)

        if args.force or not cropped_meta.exists():
            job_raw.mkdir(parents=True, exist_ok=True)
            run([
                args.blender, blend, "--background", "--python", render_script, "--",
                "--fen", job["fen"],
                "--viewpoint", job["viewpoint"],
                "--out-dir", job_raw,
                "--name", job["name"],
                "--resolution", args.resolution,
                "--samples", args.samples,
                "--camera-y", args.camera_y,
                "--camera-z", cam_z,
                "--lens", args.lens,
            ], quiet=args.quiet)
            job_crop.mkdir(parents=True, exist_ok=True)
            run([
                sys.executable, crop_script,
                "--metadata", meta,
                "--out-dir", job_crop,
                "--size", "512",
            ], quiet=args.quiet)

        if args.force or not pair_out.exists():
            real = real_target_from_v4(v4_dataset, job)
            build_pair(job_crop / f"{job['name']}_rgb.png", real, pair_out)
            shutil.copy2(job_crop / f"{job['name']}_seg.png", seg_out)
            shutil.copy2(job_crop / f"{job['name']}_depth.png", depth_out)

        out_labels[job["name"]] = {
            **job,
            "synthetic_source": str(job_crop / f"{job['name']}_rgb.png"),
            "seg_source": str(job_crop / f"{job['name']}_seg.png"),
            "depth_source": str(job_crop / f"{job['name']}_depth.png"),
            "real_source": str(v4_dataset / job["split"] / f"{job['name']}.png"),
            "render": {
                "resolution": args.resolution,
                "samples": args.samples,
                "camera_y": args.camera_y,
                "camera_z": cam_z,
                "elev_deg": game_elev.get(job["game"]),
                "lens": args.lens,
            },
        }

        existing[job["name"]] = out_labels[job["name"]]
        labels_out_path.write_text(json.dumps(existing, indent=2))

    print(f"[done] wrote {len(out_labels)} selected jobs into {out_dir}")


if __name__ == "__main__":
    main()
