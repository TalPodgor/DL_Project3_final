"""Inference helpers for the final Project 3 model `chess_v5_bright_silABC`.

This package bundles only what is needed to run the final geometry-conditioned
`paired_geom_hd` generator at submission time, with no dependency on the
training framework (the cloned CUT/pix2pixHD repo):

- `preprocess` : faithful port of the conditioning built by
  `v5_work/v5_cluster_src/v5_oblique_dataset.py` (RGB + 15-way one-hot
  silhouette segmentation + 3-channel geometry tensor).
- `networks`   : a self-contained copy of the CUT/pix2pixHD antialiased
  `resnet_9blocks` generator used by `networks.define_G(...)`, so the trained
  `latest_net_G.pth` state_dict loads without the full framework.

These are imported lazily by `generate_chessboard_image.py` (they require
torch); the synthetic-render stage does not import this package.
"""
