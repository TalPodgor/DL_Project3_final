"""Project 3 — required evaluation API: `generate_chessboard_image(fen, viewpoint)`.

Given a ground-truth FEN and a viewpoint, this generates and saves three images
to `./results/`:

    ./results/synthetic.png      clean Blender render of the position
    ./results/realistic.png      synthetic -> real translation (final GAN)
    ./results/side_by_side.png   [ synthetic | realistic ]

Final model: `chess_v5_bright_silABC` — a paired, geometry-conditioned
pix2pixHD-style translator (`paired_geom_hd`). The generator consumes the
synthetic RGB plus a 15-way one-hot silhouette segmentation and a 3-channel
geometry tensor (21 input channels) and outputs the realistic RGB.

Two stages:
  A. SYNTHETIC  — runs locally via Blender + `v5_pipeline/render_oblique_blender.py`
                  (+ `pil_perspective_crop.py`). No torch needed.
  B. REALISTIC  — needs PyTorch and the trained checkpoint at
                  `checkpoints/chess_v5_bright_silABC/latest_net_G.pth`.
                  If either is missing, a precise error is raised and NO fake
                  realistic image is written (the synthetic image is still saved).

The function is deterministic for a given (fen, viewpoint), creates `./results/`
if needed, does not return the images, and writes only inside `./results/`.
"""
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
RENDER_SCRIPT = PROJECT_ROOT / "v5_pipeline" / "render_oblique_blender.py"
CROP_SCRIPT = PROJECT_ROOT / "v5_pipeline" / "pil_perspective_crop.py"
BLEND_FILE = PROJECT_ROOT / "chess-set.blend"

MODEL_NAME = "chess_v5_bright_silABC"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / MODEL_NAME / "latest_net_G.pth"
CHECKPOINT_URL = "bundled in repo at checkpoints/chess_v5_bright_silABC/latest_net_G.pth (see submission/CHECKPOINT_MANIFEST.md)"

# Canonical "bright" oblique render parameters (from the training dataset
# labels.json for chess_v5_oblique_aligned_bright; elevation 44 deg).
RENDER_PARAMS = {
    "resolution": 900,
    "samples": 24,
    "camera_y": 1.05,
    "camera_z": 1.014,
    "lens": 48.0,
}

VALID_VIEWPOINTS = ("white", "black")


class CheckpointMissingError(FileNotFoundError):
    """Raised when the final generator checkpoint is not available locally."""


def _results_dir() -> Path:
    """`./results` relative to the current working directory (per the spec)."""
    d = Path("results")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _find_blender() -> str:
    """Locate a Blender executable (env override, PATH, or common macOS path)."""
    cand = os.environ.get("BLENDER")
    if cand and Path(cand).exists():
        return cand
    on_path = shutil.which("blender") or shutil.which("Blender")
    if on_path:
        return on_path
    for p in ("/Applications/Blender.app/Contents/MacOS/Blender",
              "/opt/homebrew/bin/blender", "/usr/local/bin/blender"):
        if Path(p).exists():
            return p
    raise RuntimeError(
        "Blender was not found. Install it from https://www.blender.org/download/ "
        "and either put `blender` on your PATH or set the BLENDER environment "
        "variable to the executable. Blender is required to render the synthetic "
        "chessboard image."
    )


def _render_synthetic(fen: str, viewpoint: str, work_dir: Path) -> dict:
    """Render rgb/seg/depth via Blender and perspective-crop to 512x512.

    Returns a dict of cropped image paths: {rgb, seg, depth}.
    """
    if not BLEND_FILE.exists():
        raise RuntimeError(f"Missing Blender asset: {BLEND_FILE}")
    if not RENDER_SCRIPT.exists() or not CROP_SCRIPT.exists():
        raise RuntimeError("Missing v5_pipeline render/crop scripts.")

    blender = _find_blender()
    raw_dir = work_dir / "raw"
    crop_dir = work_dir / "cropped"
    raw_dir.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)
    name = "sample"

    subprocess.run([
        blender, str(BLEND_FILE), "--background", "--python", str(RENDER_SCRIPT), "--",
        "--fen", fen,
        "--viewpoint", viewpoint,
        "--out-dir", str(raw_dir),
        "--name", name,
        "--resolution", str(RENDER_PARAMS["resolution"]),
        "--samples", str(RENDER_PARAMS["samples"]),
        "--camera-y", str(RENDER_PARAMS["camera_y"]),
        "--camera-z", str(RENDER_PARAMS["camera_z"]),
        "--lens", str(RENDER_PARAMS["lens"]),
    ], check=True)

    metadata = raw_dir / f"{name}_metadata.json"
    if not metadata.exists():
        raise RuntimeError(f"Blender render did not produce {metadata}")

    subprocess.run([
        sys.executable, str(CROP_SCRIPT),
        "--metadata", str(metadata),
        "--out-dir", str(crop_dir),
        "--size", "512",
    ], check=True)

    out = {k: crop_dir / f"{name}_{k}.png" for k in ("rgb", "seg", "depth")}
    for k, p in out.items():
        if not p.exists():
            raise RuntimeError(f"Cropped {k} image missing: {p}")
    return out


def _load_generator(checkpoint_path: Path):
    """Build the vendored generator and load the trained state_dict (strict)."""
    import torch  # local import: only needed for the realistic stage

    from chess_v5_infer.networks import build_generator

    net = build_generator(in_ch=21, out_ch=3, ngf=64, n_blocks=9)
    state_dict = torch.load(str(checkpoint_path), map_location="cpu")
    if hasattr(state_dict, "_metadata"):
        del state_dict._metadata
    missing, unexpected = net.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Generator state_dict did not match the bundled architecture.\n"
            f"  missing keys ({len(missing)}): {list(missing)[:6]}...\n"
            f"  unexpected keys ({len(unexpected)}): {list(unexpected)[:6]}...\n"
            "If you have the cloned CUT framework on PYTHONPATH, load via its "
            "`networks.define_G(21, 3, 64, 'resnet_9blocks', 'instance', False, "
            "'xavier', 0.02, False, False, [], opt)` instead."
        )
    net.eval()
    return net


def _to_uint8_image(tensor):
    """Denormalize a [1,3,H,W] tensor in [-1,1] to a PIL RGB image."""
    import numpy as np
    from PIL import Image

    arr = tensor.detach().squeeze(0).clamp(-1, 1).permute(1, 2, 0).cpu().numpy()
    arr = ((arr + 1.0) * 127.5).round().clip(0, 255).astype("uint8")
    return Image.fromarray(arr, mode="RGB")


def _make_realistic(synthetic_paths: dict, fen: str, viewpoint: str):
    """Run the final generator. Returns a PIL realistic image.

    Raises CheckpointMissingError / RuntimeError (never fabricates output).
    """
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for the realistic translation stage but is not "
            "installed. Install it (e.g. `pip install torch torchvision`) or run "
            "in the training environment (the cluster `pytorch` conda env), then "
            "re-run. The synthetic image was still written to ./results/synthetic.png."
        ) from exc

    if not CHECKPOINT_PATH.exists():
        raise CheckpointMissingError(
            "Final generator checkpoint not found.\n"
            f"  Expected: {CHECKPOINT_PATH}\n"
            f"  Model:    {MODEL_NAME} (paired_geom_hd)\n"
            f"  Download: {CHECKPOINT_URL}\n"
            "Place `latest_net_G.pth` at the expected path and re-run. The "
            "synthetic image was still written to ./results/synthetic.png. "
            "See submission/CHECKPOINT_MANIFEST.md for details."
        )

    import torch
    from chess_v5_infer.preprocess import build_inputs

    A, seg, geom = build_inputs(
        synthetic_paths["rgb"], synthetic_paths["seg"], synthetic_paths["depth"],
        fen, viewpoint,
    )
    net = _load_generator(CHECKPOINT_PATH)
    with torch.no_grad():
        onehot = torch.nn.functional.one_hot(seg.clamp(0, 14), 15).permute(0, 3, 1, 2).float()
        gen_input = torch.cat([A, onehot, geom], dim=1)
        fake = net(gen_input)
    return _to_uint8_image(fake)


def generate_chessboard_image(fen: str, viewpoint: str) -> None:
    """Generate synthetic + realistic chessboard images from a FEN.

    Args:
        fen: a FEN string (same conventions as the Project 1/2 ground-truth CSVs).
        viewpoint: "white" or "black" (which color is closest to the camera).

    Saves ./results/{synthetic,realistic,side_by_side}.png. Returns None.
    """
    if not isinstance(fen, str) or not fen.strip():
        raise ValueError("`fen` must be a non-empty FEN string.")
    if viewpoint not in VALID_VIEWPOINTS:
        raise ValueError(f"`viewpoint` must be one of {VALID_VIEWPOINTS}, got {viewpoint!r}.")

    from PIL import Image

    results = _results_dir()
    synthetic_out = results / "synthetic.png"
    realistic_out = results / "realistic.png"
    side_by_side_out = results / "side_by_side.png"

    with tempfile.TemporaryDirectory(prefix="chess_v5_gen_") as tmp:
        work = Path(tmp)
        # Stage A: synthetic (local Blender render). Always produced.
        synthetic_paths = _render_synthetic(fen, viewpoint, work)
        synthetic_img = Image.open(synthetic_paths["rgb"]).convert("RGB").resize((512, 512), Image.BICUBIC)
        synthetic_img.save(synthetic_out)

        # Stage B: realistic (GAN). Raises cleanly if torch/checkpoint missing.
        realistic_img = _make_realistic(synthetic_paths, fen, viewpoint)
        realistic_img = realistic_img.resize((512, 512), Image.BICUBIC)
        realistic_img.save(realistic_out)

        # Side-by-side: [ synthetic | realistic ].
        canvas = Image.new("RGB", (1024, 512))
        canvas.paste(synthetic_img, (0, 0))
        canvas.paste(realistic_img, (512, 0))
        canvas.save(side_by_side_out)


if __name__ == "__main__":
    # Minimal self-run on the standard starting position, white viewpoint.
    generate_chessboard_image(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "white")
    print("Wrote ./results/synthetic.png, ./results/realistic.png, ./results/side_by_side.png")
