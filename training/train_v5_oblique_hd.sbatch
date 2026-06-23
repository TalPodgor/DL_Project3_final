#!/bin/bash
#SBATCH --job-name=chess_v5obl
#SBATCH --partition=gtx1080
#SBATCH --qos=course
#SBATCH --gres=gpu:1
#SBATCH --time=0-23:30:00
#SBATCH --output=/home/packler/chess_cut_project/v5_oblique_log_%j.txt

set -euo pipefail

# V5A training ablation.
# Input = oblique Blender RGB + recovered semantic silhouettes + depth/silhouette/edge geometry.
# Dataset must be middle-only; do not train left/right samples against the same real target.
#
# Env overrides:
#   DATAROOT=./datasets/chess_v5_oblique NAME=chess_v5_oblique_a BS=2 EP=30 EPD=20
#   PIECEW=0.35 PCLS=0.5 INIT=./checkpoints/chess_geomv4_smoke5/latest_net_G.pth RESUME=1
#   Local piece-crop objective: PGAN=1.0 PFM=10 PVGG=5 PCROPS=64 PCROPSZ=96 VGG=3

cd ~/chess_cut_project/contrastive-unpaired-translation
echo '=== GPU ===' && nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv && date

DATAROOT=${DATAROOT:-./datasets/chess_v5_oblique}
NAME=${NAME:-chess_v5_oblique_a}
BS=${BS:-2}
EP=${EP:-30}
EPD=${EPD:-20}
PIECEW=${PIECEW:-0.35}
PCLS=${PCLS:-0.5}
INIT=${INIT:-./checkpoints/chess_geomv4_smoke5/latest_net_G.pth}
# Global VGG down-weighted (was 5.0) to reduce averaging; local piece-crop objective:
VGG=${VGG:-3.0}
PGAN=${PGAN:-1.0}
PFM=${PFM:-10.0}
PVGG=${PVGG:-5.0}
PCX=${PCX:-0.0}      # crop-level Contextual Loss weight (anti intra-piece double-head)
PSHAPE=${PSHAPE:-0.0} # crop-level pseudo-mask shape/topology loss weight
PSHAPESZ=${PSHAPESZ:-48}
PSHAPETH=${PSHAPETH:-0.10}
PSHAPETEMP=${PSHAPETEMP:-0.035}
PSHAPEOUT=${PSHAPEOUT:-1.5}
PSRCSHAPE=${PSRCSHAPE:-0.0}      # source-silhouette crop shape/topology guardrail
PSRCSHAPEIN=${PSRCSHAPEIN:-0.25}
PSRCSHAPEOUT=${PSRCSHAPEOUT:-3.0}
PSRCSHAPEBLUR=${PSRCSHAPEBLUR:-1}
PCROPS=${PCROPS:-64}
PCROPSZ=${PCROPSZ:-96}
CX=${CX:-0.0}        # Contextual Loss weight (misalignment-robust); 0 = off
CXLVL=${CXLVL:-2}    # VGG slice for CX features (~relu3)
CXSZ=${CXSZ:-48}     # CX feature map HxW (smaller = less memory)
VGGMAX=${VGGMAX:-256}  # VGG input cap (smaller = less memory)
MAXDS=${MAXDS:-0}   # >0 limits dataset size (smoke runs)
PRINTF=${PRINTF:-100}
SEMSRC=${SEMSRC:-fen}   # Fix A: fen | silhouette | fen_silhouette (silhouette-shaped seg)
EDGE=${EDGE:-0.0}       # Fix B: Sobel boundary loss weight on the silhouette edge band
HEADW=${HEADW:-0.0}     # Fix F: extra L1 on top-of-piece (anti-vanishing-head)
SHADOWW=${SHADOWW:-0.0} # Fix F: extra L1 on below-base band (contact-shadow grounding)
HEADPX=${HEADPX:-20}    # rows from silhouette top counted as head
GLOCK=${GLOCK:-0.0}     # geometry-lock alpha for piece regions (0=off)
GLOCKFEATHER=${GLOCKFEATHER:-0}
GLOCKBLUR=${GLOCKBLUR:-0}
GLOCKCONTRAST=${GLOCKCONTRAST:-0.45}
PCOMP=${PCOMP:-0.0}     # foreground/background two-stream composite alpha (0=off)
PCOMPFEATHER=${PCOMPFEATHER:-1}

DS_ARGS=""
if [ "$MAXDS" != "0" ]; then
  DS_ARGS="--max_dataset_size $MAXDS"
  echo "SMOKE: limiting dataset to $MAXDS samples"
fi

if [ ! -d "$DATAROOT/train" ]; then
  echo "Missing train split: $DATAROOT/train"
  exit 2
fi

if [ ! -f "$INIT" ]; then
  INIT=./checkpoints/chess_geomv4_smoke/latest_net_G.pth
fi
if [ ! -f "$INIT" ]; then
  INIT=./checkpoints/chess_paired/latest_net_G.pth
fi

INIT_ARGS="--epoch_count 1"
if [ -f "$INIT" ]; then
  INIT_ARGS="--g_init_path $INIT --epoch_count 1"
  echo "Warm-starting G from $INIT"
else
  echo "No compatible warm-start checkpoint found; training G from scratch"
fi

if [ "${RESUME:-0}" = "1" ]; then
  INIT_ARGS="--continue_train --epoch latest"
  echo "RESUMING from checkpoints/$NAME/latest_net_{G,D}.pth"
fi

PCLS_ARGS="--lambda_piece_cls $PCLS --piece_cls_path ./checkpoints/square_eval.pth"
if [ "$PCLS" != "0" ] && [ "$PCLS" != "0.0" ] && [ ! -f ./checkpoints/square_eval.pth ]; then
  echo "Missing ./checkpoints/square_eval.pth but PCLS=$PCLS"
  exit 2
fi

conda run --no-capture-output -n pytorch python -u train.py \
  --dataroot "$DATAROOT" \
  --name "$NAME" \
  --model paired_geom_hd \
  --dataset_mode v5_oblique \
  --direction AtoB \
  --input_nc 3 --output_nc 3 --num_seg_classes 15 --num_geom_channels 3 \
  --load_size 512 --crop_size 512 \
  --batch_size "$BS" \
  --netG resnet_9blocks --ngf 64 --normG instance --no_dropout \
  --n_layers_D 3 --ndf 64 --num_D 2 --gan_mode lsgan \
  --lambda_GAN 1.0 --lambda_feat 10.0 --lambda_VGG "$VGG" --lambda_L1 7.5 \
  --l1_piece_w "$PIECEW" --lambda_edge "$EDGE" --vgg_max "$VGGMAX" \
  --l1_head_w "$HEADW" --l1_shadow_w "$SHADOWW" --head_px "$HEADPX" \
  --geom_lock_alpha "$GLOCK" --geom_lock_feather "$GLOCKFEATHER" \
  --geom_lock_blur "$GLOCKBLUR" --geom_lock_contrast "$GLOCKCONTRAST" \
  --piece_comp_alpha "$PCOMP" --piece_comp_feather "$PCOMPFEATHER" \
  --v5_semantic_source "$SEMSRC" \
  --lambda_piece_gan "$PGAN" --lambda_piece_fm "$PFM" --lambda_piece_vgg "$PVGG" \
  --lambda_piece_cx "$PCX" \
  --lambda_piece_shape "$PSHAPE" --piece_shape_size "$PSHAPESZ" \
  --piece_shape_thresh "$PSHAPETH" --piece_shape_temp "$PSHAPETEMP" \
  --piece_shape_out_w "$PSHAPEOUT" \
  --lambda_piece_src_shape "$PSRCSHAPE" --piece_src_shape_in_w "$PSRCSHAPEIN" \
  --piece_src_shape_out_w "$PSRCSHAPEOUT" --piece_src_shape_blur "$PSRCSHAPEBLUR" \
  --piece_crops_max "$PCROPS" --piece_crop_size "$PCROPSZ" --piece_class_balance 1 \
  --lambda_cx "$CX" --cx_level "$CXLVL" --cx_feat_size "$CXSZ" \
  $PCLS_ARGS \
  --lr 0.0002 --beta1 0.5 \
  --n_epochs "$EP" --n_epochs_decay "$EPD" \
  --gpu_ids 0 --display_id -1 \
  --save_epoch_freq 5 --save_latest_freq 5000 --print_freq "$PRINTF" \
  --checkpoints_dir ./checkpoints \
  $DS_ARGS \
  $INIT_ARGS

echo '=== TRAIN DONE ===' && date
