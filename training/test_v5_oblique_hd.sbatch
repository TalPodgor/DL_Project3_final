#!/bin/bash
#SBATCH --job-name=v5obl_test
#SBATCH --partition=gtx1080
#SBATCH --qos=course
#SBATCH --gres=gpu:1
#SBATCH --time=0-00:25:00
#SBATCH --output=/home/packler/chess_cut_project/v5_oblique_test_log_%j.txt

set -euo pipefail

# Usage: sbatch test_v5_oblique_hd.sbatch [EPOCH] [NAME]
# Env: DATAROOT, NUM_TEST, SEMSRC (must match training: fen|silhouette|fen_silhouette)

EPOCH=${1:-latest}
NAME=${2:-chess_v5_oblique_a}
DATAROOT=${DATAROOT:-./datasets/chess_v5_oblique}
NUM_TEST=${NUM_TEST:-140}
SEMSRC=${SEMSRC:-fen}
GLOCK=${GLOCK:-0.0}
GLOCKFEATHER=${GLOCKFEATHER:-0}
GLOCKBLUR=${GLOCKBLUR:-0}
GLOCKCONTRAST=${GLOCKCONTRAST:-0.45}
PCOMP=${PCOMP:-0.0}
PCOMPFEATHER=${PCOMPFEATHER:-1}

cd ~/chess_cut_project/contrastive-unpaired-translation

conda run -n pytorch python test.py \
  --dataroot "$DATAROOT" \
  --name "$NAME" \
  --model paired_geom_hd \
  --dataset_mode v5_oblique \
  --direction AtoB \
  --input_nc 3 --output_nc 3 --num_seg_classes 15 --num_geom_channels 3 \
  --netG resnet_9blocks --ngf 64 --normG instance --no_dropout \
  --load_size 512 --crop_size 512 \
  --v5_semantic_source "$SEMSRC" \
  --geom_lock_alpha "$GLOCK" --geom_lock_feather "$GLOCKFEATHER" \
  --geom_lock_blur "$GLOCKBLUR" --geom_lock_contrast "$GLOCKCONTRAST" \
  --piece_comp_alpha "$PCOMP" --piece_comp_feather "$PCOMPFEATHER" \
  --phase test --epoch "$EPOCH" \
  --gpu_ids 0 --num_test "$NUM_TEST" \
  --results_dir ./results

echo "done epoch=$EPOCH name=$NAME dataroot=$DATAROOT semsrc=$SEMSRC glock=$GLOCK pcomp=$PCOMP"
