"""Build the `paired_geom_hd` generator inputs from a single synthetic render.

This is a faithful, framework-free port of the conditioning produced by
`v5_work/v5_cluster_src/v5_oblique_dataset.py` (the `v5_oblique` dataset used to
train `chess_v5_bright_silABC`). At inference the model receives exactly three
tensors derived from the FEN + the Blender synthetic render alone:

  A    : synthetic RGB,                    [1, 3, 512, 512] in [-1, 1]
  seg  : semantic silhouette ids (0..14),  [1, 512, 512]    int64
  geom : depth + silhouette + edge,        [1, 3, 512, 512] in [-1, 1]

`v5_semantic_source` is `fen_silhouette` for the final model, so `seg` combines
the per-cell FEN identity with the rendered piece silhouette. The geometry
silhouette/edge are derived from the raw silhouette ids only. All of this is
"inference-safe": nothing here looks at the real/target image.
"""
import numpy as np
import torch
from PIL import Image

# --- class encoding (identical to v5_oblique_dataset.py) -------------------
PIECE_TO_ID = {
    "P": 3, "N": 4, "B": 5, "R": 6, "Q": 7, "K": 8,
    "p": 9, "n": 10, "b": 11, "r": 12, "q": 13, "k": 14,
}

CLASS_ID = {"empty_light": 1, "empty_dark": 2}
for _i, _kind in enumerate(["p", "n", "b", "r", "q", "k"]):
    CLASS_ID["w" + _kind] = 3 + _i
    CLASS_ID["b" + _kind] = 9 + _i

CLASS_COLORS_LINEAR = {
    "P": (0.92, 0.88, 0.72), "N": (0.82, 0.82, 0.52), "B": (0.92, 0.72, 0.44),
    "R": (0.72, 0.92, 0.72), "Q": (0.72, 0.82, 0.98), "K": (0.90, 0.70, 0.98),
    "p": (0.18, 0.12, 0.08), "n": (0.32, 0.18, 0.08), "b": (0.20, 0.32, 0.12),
    "r": (0.10, 0.24, 0.34), "q": (0.34, 0.12, 0.30), "k": (0.34, 0.10, 0.10),
}

TILE = 512


def linear_to_srgb_u8(v):
    v = np.asarray(v, dtype=np.float32)
    srgb = np.where(v <= 0.0031308, 12.92 * v, 1.055 * np.power(v, 1.0 / 2.4) - 0.055)
    return np.clip(np.round(srgb * 255.0), 0, 255).astype(np.int16)


PALETTE = np.stack([linear_to_srgb_u8(CLASS_COLORS_LINEAR[p]) for p in PIECE_TO_ID], axis=0)
PALETTE_IDS = np.asarray([PIECE_TO_ID[p] for p in PIECE_TO_ID], dtype=np.uint8)


def rgb_semantic_to_ids(seg_rgb, threshold=75, bright=30):
    """Map anti-aliased Blender semantic colors to class ids (0, 3..14)."""
    arr = np.asarray(seg_rgb.convert("RGB"), dtype=np.int16)
    flat = arr.reshape(-1, 3)
    diff = flat[:, None, :] - PALETTE[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    nearest = np.argmin(dist2, axis=1)
    best = dist2[np.arange(flat.shape[0]), nearest]
    ids = np.zeros(flat.shape[0], dtype=np.uint8)
    bright_enough = np.max(flat, axis=1) >= bright
    keep = (best <= threshold * threshold) & bright_enough
    ids[keep] = PALETTE_IDS[nearest[keep]]
    return ids.reshape(arr.shape[:2])


def fen_piece_grid(fen, viewpoint):
    ids = np.zeros((8, 8), np.uint8)
    rows = fen.split()[0].split("/")
    for r, row in enumerate(rows):
        c = 0
        for ch in row:
            if ch.isdigit():
                c += int(ch)
                continue
            color = "w" if ch.isupper() else "b"
            ids[r, c] = CLASS_ID[color + ch.lower()]
            c += 1
    if viewpoint == "black":
        ids = np.rot90(ids, 2)
    return ids


def fen_silhouette_seg(fen, viewpoint, sil_ids, tile=TILE, base_floor=0.04):
    """Silhouette-shaped semantic conditioning with a per-cell FEN identity floor.

    1. empty-board ids (light/dark by parity) fill the whole board.
    2. the per-pixel render id (sil_ids) paints the visible piece silhouette.
    3. FEN base-patch guardrail: every FEN-occupied cell keeps >= a small footprint
       of its own id in the base region, so occluded/tiny pieces keep their identity.
    """
    grid = fen_piece_grid(fen, viewpoint)
    cell = tile // 8
    seg = np.zeros((tile, tile), np.uint8)
    for r in range(8):
        for c in range(8):
            seg[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] = (
                CLASS_ID["empty_light"] if ((r + c) % 2 == 0) else CLASS_ID["empty_dark"])
    pm = sil_ids >= 3
    seg[pm] = sil_ids[pm].astype(np.uint8)
    bh0 = int(cell * 0.45)
    bx0, bx1 = int(cell * 0.15), int(cell * 0.85)
    for r in range(8):
        for c in range(8):
            fid = int(grid[r, c])
            if fid == 0:
                continue
            blk = seg[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell]
            if (blk == fid).mean() >= base_floor:
                continue
            sub = blk[bh0:cell, bx0:bx1]
            free = (sub < 3) | (sub == fid)
            sub[free] = fid
    return seg


def silhouette_edge(piece):
    edge = torch.zeros_like(piece)
    edge[1:, :] = torch.maximum(edge[1:, :], (piece[1:, :] != piece[:-1, :]).float())
    edge[:-1, :] = torch.maximum(edge[:-1, :], (piece[:-1, :] != piece[1:, :]).float())
    edge[:, 1:] = torch.maximum(edge[:, 1:], (piece[:, 1:] != piece[:, :-1]).float())
    edge[:, :-1] = torch.maximum(edge[:, :-1], (piece[:, :-1] != piece[:, 1:]).float())
    return edge


def _rgb_to_tensor(img):
    arr = np.asarray(img.convert("RGB"), np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def _gray_to_tensor(img):
    arr = np.asarray(img.convert("L"), np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)


def _ids_to_tensor(ids, tile=TILE):
    seg = Image.fromarray(ids).resize((tile, tile), Image.NEAREST)
    return torch.from_numpy(np.array(seg, dtype=np.int64)).unsqueeze(0)


def build_inputs(rgb_path, seg_path, depth_path, fen, viewpoint,
                 seg_color_threshold=75, seg_bright_gate=30, tile=TILE):
    """Return (A, seg, geom) batched tensors for the paired_geom_hd generator.

    Mirrors `V5ObliqueDataset.__getitem__` for the `fen_silhouette` semantic
    source with no augmentation (deterministic inference).
    """
    rgb = Image.open(rgb_path).convert("RGB").resize((tile, tile), Image.BICUBIC)
    seg_rgb = Image.open(seg_path).convert("RGB")
    depth = Image.open(depth_path).convert("L")

    sil_ids = rgb_semantic_to_ids(seg_rgb, seg_color_threshold, seg_bright_gate)
    seg_ids = fen_silhouette_seg(fen, viewpoint, sil_ids, tile)

    A = _rgb_to_tensor(rgb)
    seg = _ids_to_tensor(seg_ids, tile)

    depth_t = _gray_to_tensor(depth.resize((tile, tile), Image.BILINEAR))[0]   # [1,H,W]
    silhouette_seg = _ids_to_tensor(sil_ids, tile)[0]                          # [H,W]
    piece = (silhouette_seg >= 3).float()
    edge = silhouette_edge(piece)
    geom = torch.cat([
        depth_t,
        (piece * 2.0 - 1.0).unsqueeze(0),
        (edge * 2.0 - 1.0).unsqueeze(0),
    ], dim=0).unsqueeze(0)
    return A, seg, geom
