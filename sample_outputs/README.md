# Sample outputs

These are **real** outputs produced by this repository's code on this machine.

| File | What it is |
|---|---|
| `synthetic_white_view.png` | Stage A synthetic render, `viewpoint="white"` (white pieces closest to camera), standard starting FEN. |
| `synthetic_black_view.png` | Stage A synthetic render, `viewpoint="black"` (black pieces closest to camera), standard starting FEN. |

Both were generated with:

```bash
python run_project3_demo.py --viewpoint white   # -> ./results/synthetic.png
python run_project3_demo.py --viewpoint black   # -> ./results/synthetic.png
```

(512×512 RGB PNG, oblique 44° "bright" render — the same synthetic distribution
the final model was trained on.)

## Why there is no `realistic.png` / `side_by_side.png` here

The realistic (GAN) stage needs **PyTorch** and the trained checkpoint
`checkpoints/chess_v5_bright_silABC/latest_net_G.pth`, which are **not present on
the machine that prepared this submission** (the model was trained on the BGU
cluster; the weights live on the shared drive — see `../CHECKPOINT_MANIFEST.md`).

Per the submission guidance, **no fabricated realistic image is included.** On a
machine with PyTorch and the checkpoint in place, the same command produces all
three files in `./results/`:

```
./results/synthetic.png
./results/realistic.png
./results/side_by_side.png
```
