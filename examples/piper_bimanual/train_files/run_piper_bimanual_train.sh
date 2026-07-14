#!/usr/bin/env bash
set -euo pipefail

config_yaml=./examples/piper_bimanual/train_files/starvla_finetune_piper_bimanual.yaml
run_root_dir=./results/Checkpoints
run_id=starvla_piper_bimanual
pretrain_ckpt=./checkpoints/Multi-view-VLA/pretrained_model/checkpoints/steps_14000_pytorch_model.pt
freeze_modules=${FREEZE_MODULES:-spatial_model,image_edit_model}
# Set once here, or override at launch:
#   GPUS=0,3,4,7 bash examples/piper_bimanual/train_files/run_piper_bimanual_train.sh
#   GPUS=7 bash examples/piper_bimanual/train_files/run_piper_bimanual_train.sh
GPUS=${GPUS:-0,1,2,3,4,5,6,7}

export WANDB_MODE=disabled
export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export NO_ALBUMENTATIONS_UPDATE=1
export CUDA_VISIBLE_DEVICES="${GPUS}"

num_processes=$(python - <<'PY'
import os
gpus = [gpu.strip() for gpu in os.environ["CUDA_VISIBLE_DEVICES"].split(",") if gpu.strip()]
if not gpus:
    raise SystemExit("GPUS/CUDA_VISIBLE_DEVICES must contain at least one GPU id")
print(len(gpus))
PY
)

echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} with ${num_processes} process(es)"
echo "Freezing modules: ${freeze_modules}"

accelerate launch \
  --num_processes "${num_processes}" \
  --mixed_precision bf16 \
  starVLA/training/train_starvla.py \
  --config_yaml "${config_yaml}" \
  --datasets.vla_data.data_root_dir ./datasets \
  --datasets.vla_data.data_mix piper_bimanual_stack_cup_bowl \
  --datasets.vla_data.video_backend torchvision_av \
  --trainer.pretrained_checkpoint "${pretrain_ckpt}" \
  --trainer.reload_modules qwen_vl_interface,action_model \
  --trainer.freeze_modules "${freeze_modules}" \
  --run_root_dir "${run_root_dir}" \
  --run_id "${run_id}"
