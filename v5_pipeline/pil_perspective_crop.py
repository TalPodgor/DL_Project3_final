"""
Perspective-crop V5 Blender raw renders using only PIL/numpy.

The Blender renderer writes board corner coordinates in raw image coordinates.
This script warps RGB/seg/depth raw renders so the board plane fills a square.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def perspective_coeffs(dst_points, src_points):
    matrix = []
    vector = []
    for (x, y), (u, v) in zip(dst_points, src_points):
        matrix.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        matrix.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        vector.extend([u, v])
    return np.linalg.solve(np.asarray(matrix, dtype=np.float64), np.asarray(vector, dtype=np.float64))


def crop_one(src, dst, corners, size, resample):
    img = Image.open(src).convert("RGB")
    dst_points = [(0, 0), (size - 1, 0), (size - 1, size - 1), (0, size - 1)]
    coeffs = perspective_coeffs(dst_points, corners)
    warped = img.transform((size, size), Image.Transform.PERSPECTIVE, coeffs, resample=resample)
    warped.save(dst)
    return warped


def make_montage(out_dir, name, images):
    label_h = 24
    thumb = 256
    labels = ["rgb", "seg", "depth"]
    canvas = Image.new("RGB", (thumb * len(labels), thumb + label_h), "white")
    draw = ImageDraw.Draw(canvas)
    for i, label in enumerate(labels):
        im = images[label].resize((thumb, thumb), Image.Resampling.LANCZOS)
        x = i * thumb
        canvas.paste(im, (x, label_h))
        draw.text((x + 6, 5), label, fill=(0, 0, 0))
    path = out_dir / f"{name}_cropped_montage.png"
    canvas.save(path)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--size", type=int, default=512)
    args = ap.parse_args()

    meta = json.loads(Path(args.metadata).read_text())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = meta["name"]
    corners = [tuple(p) for p in meta["board_corners_2d"]]

    images = {
        "rgb": crop_one(meta["raw"]["rgb"], out_dir / f"{name}_rgb.png", corners, args.size, Image.Resampling.BICUBIC),
        "seg": crop_one(meta["raw"]["seg"], out_dir / f"{name}_seg.png", corners, args.size, Image.Resampling.NEAREST),
        "depth": crop_one(meta["raw"]["depth"], out_dir / f"{name}_depth.png", corners, args.size, Image.Resampling.BILINEAR),
    }
    montage = make_montage(out_dir, name, images)

    cropped_meta = dict(meta)
    cropped_meta["cropped"] = {k: str(out_dir / f"{name}_{k}.png") for k in images}
    cropped_meta["cropped_montage"] = str(montage)
    (out_dir / f"{name}_cropped_metadata.json").write_text(json.dumps(cropped_meta, indent=2))
    print(montage)


if __name__ == "__main__":
    main()
