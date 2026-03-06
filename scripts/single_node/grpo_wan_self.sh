#!/bin/bash
# WAN 2.1 SOLACE Training (single node)
# Usage:
#   8 GPU:  bash scripts/single_node/grpo_wan_self.sh

accelerate launch \
    --config_file scripts/accelerate_configs/multi_gpu.yaml \
    --num_processes=8 \
    --main_process_port 29503 \
    scripts/train_wan2_1_self.py \
    --config config/solace.py:general_ocr_wan2_1
