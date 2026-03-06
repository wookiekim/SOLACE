# 8 GPU
accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=8 --main_process_port 29501 scripts/train_flux_self_accelerate.py --config config/solace.py:general_ocr_flux_8gpu
