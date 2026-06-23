"""V5 oblique geometry-conditioned aligned dataset.

Each item:
  {name}.png        = [A_v5_oblique_rgb | B_real]
  {name}_seg.png    = RGB semantic silhouette render from Blender
  {name}_depth.png  = grayscale depth-like render from Blender

The model still receives the same contract as paired_geom_hd:
  A RGB, seg ids, geom tensor.

For V5, seg ids default to full-cell FEN semantics from labels.json. The
Blender semantic colors are still used for geometry, so the model gets both:
explicit empty/occupied square labels and true rendered piece silhouettes.

Geom is:
  channel 0: depth render
  channel 1: rendered piece silhouette
  channel 2: silhouette boundary
"""
import json
import os

import numpy as np
import torch
from PIL import Image

from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset


PIECE_TO_ID = {
    "P": 3, "N": 4, "B": 5, "R": 6, "Q": 7, "K": 8,
    "p": 9, "n": 10, "b": 11, "r": 12, "q": 13, "k": 14,
}

CLASS_ID = {"empty_light": 1, "empty_dark": 2}
for i, kind in enumerate(["p", "n", "b", "r", "q", "k"]):
    CLASS_ID["w" + kind] = 3 + i
    CLASS_ID["b" + kind] = 9 + i

CLASS_COLORS_LINEAR = {
    "P": (0.92, 0.88, 0.72),
    "N": (0.82, 0.82, 0.52),
    "B": (0.92, 0.72, 0.44),
    "R": (0.72, 0.92, 0.72),
    "Q": (0.72, 0.82, 0.98),
    "K": (0.90, 0.70, 0.98),
    "p": (0.18, 0.12, 0.08),
    "n": (0.32, 0.18, 0.08),
    "b": (0.20, 0.32, 0.12),
    "r": (0.10, 0.24, 0.34),
    "q": (0.34, 0.12, 0.30),
    "k": (0.34, 0.10, 0.10),
}


def linear_to_srgb_u8(v):
    v = np.asarray(v, dtype=np.float32)
    srgb = np.where(v <= 0.0031308, 12.92 * v, 1.055 * np.power(v, 1.0 / 2.4) - 0.055)
    return np.clip(np.round(srgb * 255.0), 0, 255).astype(np.int16)


PALETTE = np.stack([linear_to_srgb_u8(CLASS_COLORS_LINEAR[p]) for p in PIECE_TO_ID], axis=0)
PALETTE_IDS = np.asarray([PIECE_TO_ID[p] for p in PIECE_TO_ID], dtype=np.uint8)


def rgb_semantic_to_ids(seg_rgb, threshold=75, bright=30):
    """Map anti-aliased Blender semantic colors to class ids.

    Dark board/background/frame pixels remain 0. Piece pixels are assigned to the
    nearest known emitted color if they are close enough in RGB space.

    Fix D: the brightness gate was 70, which dropped the dark anti-aliased EDGE
    pixels of BLACK pieces (their emitted colors are darker, so half-blended edge
    pixels fall under 70) -> black silhouettes were systematically under-captured.
    The semantic render has a pure-black background (only pieces get class colors),
    and the color-distance gate (best <= threshold^2) already rejects true black,
    so the brightness gate can safely drop to 30 to recover those edge pixels.
    """
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


def full_cell_seg_from_fen(fen, viewpoint, tile):
    grid = fen_piece_grid(fen, viewpoint)
    seg = np.zeros((tile, tile), np.uint8)
    cell = tile // 8
    for r in range(8):
        for c in range(8):
            cid = int(grid[r, c])
            if cid == 0:
                cid = CLASS_ID["empty_light"] if ((r + c) % 2 == 0) else CLASS_ID["empty_dark"]
            seg[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] = cid
    return seg


def fen_silhouette_seg(fen, viewpoint, sil_ids, tile, base_floor=0.04):
    """Fix A: silhouette-shaped semantic conditioning (kills square halos / wood
    bleed) while keeping a robust per-cell identity floor (protects occupancy).

      1. empty-board ids (light/dark by parity) fill the whole board.
      2. the per-pixel render id (sil_ids, from rgb_semantic_to_ids) paints the
         VISIBLE piece silhouette -- this is overhang-correct: tall leaning pieces
         (~2 cells tall) spill into neighbour cells and that material keeps the
         correct piece id, which cell-quantised FEN masking would wrongly drop.
      3. FEN base-patch guardrail: every FEN-occupied cell is guaranteed >= a small
         footprint of its OWN id in the base region, so a piece whose visible
         silhouette is tiny/occluded never loses its identity/occupancy signal.

    Inference-safe: both the FEN grid and sil_ids derive from the FEN/synthetic
    render alone (no real/temporal input).
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
    bh0 = int(cell * 0.45)            # base region = bottom 55% of the cell
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
            free = (sub < 3) | (sub == fid)   # don't clobber a different visible piece
            sub[free] = fid
    return seg


def silhouette_edge(piece):
    edge = torch.zeros_like(piece)
    edge[1:, :] = torch.maximum(edge[1:, :], (piece[1:, :] != piece[:-1, :]).float())
    edge[:-1, :] = torch.maximum(edge[:-1, :], (piece[:-1, :] != piece[1:, :]).float())
    edge[:, 1:] = torch.maximum(edge[:, 1:], (piece[:, 1:] != piece[:, :-1]).float())
    edge[:, :-1] = torch.maximum(edge[:, :-1], (piece[:, :-1] != piece[:, 1:]).float())
    return edge


class V5ObliqueDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument("--seg_color_threshold", type=int, default=75)
        parser.add_argument("--seg_bright_gate", type=int, default=30,
                            help="Fix D: brightness gate for silhouette pixels (was 70; "
                                 "30 recovers dark black-piece AA edges)")
        parser.add_argument("--v5_semantic_source",
                            choices=["fen", "silhouette", "fen_silhouette"], default="fen")
        return parser

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))
        self.AB_paths = [
            p for p in paths
            if not p.endswith("_seg.png") and not p.endswith("_depth.png")
        ]
        assert self.opt.load_size >= self.opt.crop_size
        self.input_nc = self.opt.input_nc
        self.output_nc = self.opt.output_nc
        labels_path = os.path.join(opt.dataroot, "labels.json")
        self.labels = json.load(open(labels_path)) if os.path.exists(labels_path) else {}

    def _ids_transform(self, ids, params):
        seg = Image.fromarray(ids)
        pp = self.opt.preprocess
        if "resize" in pp:
            seg = seg.resize((self.opt.load_size, self.opt.load_size), Image.NEAREST)
        if "crop" in pp:
            x, y = params["crop_pos"]
            cs = self.opt.crop_size
            seg = seg.crop((x, y, x + cs, y + cs))
        if (not self.opt.no_flip) and params.get("flip", False):
            seg = seg.transpose(Image.FLIP_LEFT_RIGHT)
        return torch.from_numpy(np.array(seg, dtype=np.int64))

    def _silhouette_ids(self, seg_rgb):
        gate = getattr(self.opt, "seg_bright_gate", 30)
        return rgb_semantic_to_ids(seg_rgb, self.opt.seg_color_threshold, gate)

    def _silhouette_transform(self, seg_rgb, params):
        return self._ids_transform(self._silhouette_ids(seg_rgb), params)

    def _semantic_transform(self, AB_path, seg_rgb, params):
        if self.opt.v5_semantic_source == "silhouette":
            return self._silhouette_transform(seg_rgb, params)
        name = os.path.basename(AB_path)[:-4]
        if name not in self.labels:
            raise KeyError(f"{name} missing from labels.json")
        meta = self.labels[name]
        tile = seg_rgb.size[0]
        if self.opt.v5_semantic_source == "fen_silhouette":
            sil_ids = self._silhouette_ids(seg_rgb)
            ids = fen_silhouette_seg(meta["fen"], meta["viewpoint"], sil_ids, tile)
        else:
            ids = full_cell_seg_from_fen(meta["fen"], meta["viewpoint"], tile)
        return self._ids_transform(ids, params)

    def _geom_from_depth_and_silhouette(self, depth, silhouette_seg, params):
        depth_t = get_transform(self.opt, params, grayscale=True)(depth)
        piece = (silhouette_seg >= 3).float()
        edge = silhouette_edge(piece)
        channels = [depth_t, piece.unsqueeze(0) * 2.0 - 1.0, edge.unsqueeze(0) * 2.0 - 1.0]
        geom = torch.cat(channels, dim=0)
        geom_channels = getattr(self.opt, "num_geom_channels", 3)
        if geom_channels < geom.shape[0]:
            geom = geom[:geom_channels]
        elif geom_channels > geom.shape[0]:
            pad = torch.full(
                (geom_channels - geom.shape[0], geom.shape[1], geom.shape[2]),
                -1.0,
                dtype=geom.dtype,
            )
            geom = torch.cat([geom, pad], dim=0)
        return geom

    def __getitem__(self, index):
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert("RGB")
        w, h = AB.size
        w2 = w // 2
        A = AB.crop((0, 0, w2, h))
        B = AB.crop((w2, 0, w, h))
        seg_rgb = Image.open(AB_path[:-4] + "_seg.png").convert("RGB")
        depth = Image.open(AB_path[:-4] + "_depth.png").convert("L")

        params = get_params(self.opt, A.size)
        A = get_transform(self.opt, params, grayscale=(self.input_nc == 1))(A)
        B = get_transform(self.opt, params, grayscale=(self.output_nc == 1))(B)
        seg = self._semantic_transform(AB_path, seg_rgb, params)
        silhouette_seg = self._silhouette_transform(seg_rgb, params)
        geom = self._geom_from_depth_and_silhouette(depth, silhouette_seg, params)
        return {"A": A, "B": B, "seg": seg, "geom": geom,
                "A_paths": AB_path, "B_paths": AB_path}

    def __len__(self):
        return len(self.AB_paths)
