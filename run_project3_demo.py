#!/usr/bin/env python3
"""Project 3 demo CLI.

Runs the required `generate_chessboard_image(fen, viewpoint)` function and writes
the three deliverable images to ./results/.

Examples:
    python run_project3_demo.py
    python run_project3_demo.py --fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" --viewpoint white
    python run_project3_demo.py --fen "8/8/8/8/8/8/4K3/4k3 w - - 0 1" --viewpoint black

The synthetic image is always rendered locally (Blender). The realistic image
requires PyTorch and the trained checkpoint
(checkpoints/chess_v5_bright_silABC/latest_net_G.pth). If those are missing the
demo reports exactly what to install/place and exits non-zero — it never writes
a fabricated realistic image.
"""
import argparse
import sys

from generate_chessboard_image import (
    CheckpointMissingError,
    VALID_VIEWPOINTS,
    generate_chessboard_image,
)

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate synthetic + realistic chessboard images from a FEN.")
    parser.add_argument("--fen", default=START_FEN,
                        help="FEN string (default: standard starting position).")
    parser.add_argument("--viewpoint", default="white", choices=list(VALID_VIEWPOINTS),
                        help="Which color is closest to the camera (default: white).")
    args = parser.parse_args()

    print(f"[demo] fen       = {args.fen}")
    print(f"[demo] viewpoint = {args.viewpoint}")
    try:
        generate_chessboard_image(args.fen, args.viewpoint)
    except CheckpointMissingError as exc:
        print("\n[demo] Synthetic image rendered, but the realistic stage could "
              "not run:\n", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(f"\n[demo] Could not complete the demo:\n{exc}", file=sys.stderr)
        return 1

    print("[demo] wrote ./results/synthetic.png")
    print("[demo] wrote ./results/realistic.png")
    print("[demo] wrote ./results/side_by_side.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
