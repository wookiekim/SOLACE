# SD3.5 Positive-Only Self-Confidence Training
# 1 GPU:
# accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=1 --main_process_port 29501 scripts/train_sd3_self_positive.py --config config/solace.py:general_ocr_sd3_1gpu
# 8 GPU:
accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=8 --main_process_port 29501 scripts/train_sd3_self_positive.py --config config/solace.py:general_ocr_sd3_8gpu
