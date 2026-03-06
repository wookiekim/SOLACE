#!/bin/bash
# SDXL GRPO + Self-Certainty Training (single node)
# Usage:
#   8 GPU:  bash scripts/single_node/grpo_self_sdxl.sh
#   1 GPU:  bash scripts/single_node/grpo_self_sdxl.sh 1 sdxl_self_1gpu
#   4 GPU:  bash scripts/single_node/grpo_self_sdxl.sh 4 sdxl_self_4gpu

NUM_PROCESSES=${1:-8}
CONFIG_NAME=${2:-sdxl_self_8gpu}

accelerate launch \
    --config_file scripts/accelerate_configs/multi_gpu.yaml \
    --num_processes=${NUM_PROCESSES} \
    --main_process_port 29502 \
    scripts/train_sdxl_self.py \
    --config config/solace.py:${CONFIG_NAME}
