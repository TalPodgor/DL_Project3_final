# Project 3 — Synthetic-to-Real Chessboard Image Generation from FEN

BGU *Intro to Deep Learning* — Final Project 3.

**Authors:** Tal Podgor, Ilay Vanunu, Ran Packler.

Given a chess position (**FEN**) and a **viewpoint**, the system renders a clean
**synthetic** board with Blender and then translates it into a **realistic**
photo-like image while preserving the exact piece positions and camera
orientation. It is a paired, geometry-conditioned synthetic→real image
translation model (pix2pixHD family).

---

## Required evaluation function

```python
def generate_chessboard_image(fen: str, viewpoint: str) -> None
```

`viewpoint ∈ {"white", "black"}` (which colour is closest to the camera). The
function creates `./results/` if needed, saves three PNGs, and returns nothing:

```
./results/synthetic.png       512x512 RGB   clean synthetic render of the position
./results/realistic.png       512x512 RGB   synthetic -> real translation (final model)
./results/side_by_side.png    1024x512 RGB  [ synthetic | realistic ]
```

---

## Setup (from clone to ready-to-run)

```bash
git clone https://github.com/TalPodgor/DL_Project3_final.git
cd DL_Project3_final
pip install -r requirements.txt          # numpy, Pillow, torch
```

You also need **Blender** (for the synthetic-render stage):

1. Install Blender — https://www.blender.org/download/
2. Put `blender` on your `PATH`, **or** set the `BLENDER` environment variable to
   the executable. (On macOS the app at `/Applications/Blender.app` is auto-detected.)

The trained model checkpoint is **already bundled** in this repo at
`checkpoints/chess_v5_bright_silABC/latest_net_G.pth` (~45 MB) — nothing to download.

The realistic stage runs on CPU or GPU; a GPU makes it near-instant, CPU takes a
few seconds per board.

---

## Run inference (demo)

```bash
# standard starting position, white closest to camera (defaults)
python run_project3_demo.py

# any FEN + viewpoint
python run_project3_demo.py \
  --fen "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R" \
  --viewpoint black
```

Or call the function directly:

```python
from generate_chessboard_image import generate_chessboard_image
generate_chessboard_image("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR", "white")
# -> ./results/{synthetic,realistic,side_by_side}.png
```

A FEN may be either a bare placement field (`rnbqkbnr/.../RNBQKBNR`) or a full
FEN (`... w KQkq - 0 1`) — only the placement field is used.

### Check whether a FEN is unseen by the model

```bash
python check_fen_unseen.py "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R"
```

Compares against all 876 dataset positions (train + test), so you can confirm a
test FEN was never seen during training.

---

## Final model — `chess_v5_bright_silABC`

A **paired, geometry-conditioned, pix2pixHD-style** translator. The generator is
a `resnet_9blocks` network (instance norm) that consumes a **21-channel** input
and outputs realistic RGB:

| Channels | Source |
|---|---|
| 3  | synthetic oblique RGB render |
| 15 | one-hot silhouette segmentation (empty light/dark + 12 piece classes), built from the FEN + rendered silhouette |
| 3  | geometry: depth render · piece-silhouette mask · silhouette edge |

All conditioning is **inference-safe**: at test time it is derived from the FEN +
synthetic render alone (no real/target image is used).

---

## How it works (pipeline)

1. **Render** the synthetic board with Blender (`v5_pipeline/render_oblique_blender.py`,
   using `synthetic_chess_generator.py` + `chess-set.blend`) → raw rgb / semantic /
   depth, then rectify to 512×512 (`v5_pipeline/pil_perspective_crop.py`).
2. **Build conditioning** (`chess_v5_infer/preprocess.py`): RGB + one-hot semantic +
   geometry → the 21-channel tensor.
3. **Translate** with the generator (`chess_v5_infer/networks.py`, loading
   `latest_net_G.pth`) → realistic RGB.
4. **Save** the three PNGs to `./results/`.

---

## Repository layout

```
generate_chessboard_image.py     REQUIRED API (entry point)
run_project3_demo.py             CLI wrapper (--fen / --viewpoint)
check_fen_unseen.py              verify a FEN is not in the dataset
synthetic_chess_generator.py     FEN -> Blender scene (used by the renderer)
chess-set.blend                  Blender asset (3D pieces + board)
chess_v5_infer/                  framework-free inference helpers
  ├── networks.py                resnet_9blocks generator (loads latest_net_G.pth)
  └── preprocess.py              builds the 21-channel conditioning
v5_pipeline/
  ├── render_oblique_blender.py  FEN -> oblique rgb/seg/depth (Blender)
  ├── pil_perspective_crop.py    rectify renders to 512x512
  └── build_v5_dataset.py        builds the paired training dataset
checkpoints/chess_v5_bright_silABC/latest_net_G.pth   final model (bundled)
training/                        cluster training code (reference, see below)
data/                            gt.csv (876) + gt_test.csv (140 held-out test)
sample_outputs/                  example synthetic renders
report/project3_final_report.pdf final report (PDF)
requirements.txt
```

---

## Dataset

The paired dataset (`chess_v5_oblique_aligned_bright`, 876 paired PNGs:
left = synthetic source, right = real target) lives on the shared drive:

> https://drive.google.com/drive/folders/1oJpGFstyY0AmyeIcL4Cjk1B03UtL3_cN

Labels in this repo (format `image_name, fen, view`):
- `data/gt.csv` — all 876 rows (736 train: games 4,5,6,7 · 140 test: game 2).
- `data/gt_test.csv` — the 140 held-out **test** rows (game 2 only).

The model is trained on games 4–7 and evaluated on the fully held-out **game 2**.

---

## Training (how to reproduce training)

Training was done on the BGU cluster (GPU). The model is built on top of the
cloned **CUT / pix2pix** framework
(https://github.com/taesungp/contrastive-unpaired-translation); the files in
`training/` plug into it. To reproduce training from scratch:

1. **Get the framework.** On the cluster, clone the CUT/pix2pix repo and create
   its conda env (`pytorch`), following its own README.
2. **Drop in our files** (from `training/` here):
   - `paired_geom_hd_model.py` → framework's `models/`
   - `v5_oblique_dataset.py`   → framework's `data/`
   - `square_eval.py`          → framework root (frozen square classifier used by the piece loss)
3. **Build the paired dataset** from the FENs:
   ```bash
   python v5_pipeline/build_v5_dataset.py
   # renders the synthetic side with Blender and pairs it with the real photos →
   # datasets/chess_v5_oblique_aligned_bright/{train,test}
   ```
   (The real photos and the prebuilt dataset are on the shared drive linked above.)
4. **Launch training** (uses the exact options in `bright_silABC_train_opt.txt`):
   ```bash
   sbatch training/train_v5_oblique_hd.sbatch
   ```
5. The trained generator is written to
   `checkpoints/chess_v5_bright_silABC/latest_net_G.pth` — the same checkpoint
   already bundled in this repo.

> **You do not need to retrain to run the model** — the checkpoint is bundled and
> used directly for inference.

---

## Reproducing the reported results

- **Qualitative (generated boards):** run inference (see above) on the test FENs
  in `data/gt_test.csv` — each row is an unseen **game-2** position. Use
  `check_fen_unseen.py` to confirm a FEN was never in training.
- **Quantitative (the report's square / occupancy / piece-type accuracy on the
  140 held-out game-2 boards):** evaluate with
  `training/test_v5_oblique_hd.sbatch`, which scores the generated boards with the
  frozen square classifier (`training/square_eval.py`).
- Full numbers, plots, and the ablation study are in
  [`report/project3_final_report.pdf`](report/project3_final_report.pdf).

---

## Known limitations

- The realistic stage requires PyTorch + Blender. Without them, only the
  synthetic stage runs (the function raises a precise error rather than
  fabricating a realistic image).
- Tall pieces lean across square boundaries under the oblique camera; the
  silhouette + geometry conditioning mitigates but does not fully remove
  occasional small/dark-piece legibility loss. See the report for the full
  ablation and limitations discussion.
