# SOLACE: Improving Text-to-Image Generation with Intrinsic Self-Confidence Rewards

[![CVPR](https://img.shields.io/badge/CVPR-2026-blue)](https://cvpr.thecvf.com/)
[![arXiv](https://img.shields.io/badge/arXiv-SOLACE-red)](https://arxiv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation of **SOLACE** (**S**elf-c**O**nfidence reward for a**L**igning text-to-im**A**ge models via **C**onfidenc**E** optimization).

## Abstract

SOLACE introduces intrinsic self-confidence rewards for improving text-to-image generation through reinforcement learning. Unlike prior methods that rely on external reward models (e.g., PickScore, ImageReward), SOLACE leverages the diffusion model's own denoising confidence as a training signal — requiring no additional models at training time. This approach can be used standalone or combined with external rewards for hybrid training.

## Supported Models

| Model | Type | Script |
|-------|------|--------|
| SD3.5-Medium / Large | Image | `train_sd3_self.py` |
| Flux.1-dev | Image | `train_flux_self.py` |
| SDXL | Image | `train_sdxl_self.py` |
| WAN 2.1 | Video | `train_wan2_1_self.py` |

**Training variants:**
- `train_sd3_self.py` — Self-confidence reward only
- `train_sd3_self_ext.py` — Hybrid: self-confidence + external reward
- `train_sd3_self_positive.py` — Positive-only self-confidence

## Installation

```bash
git clone https://github.com/wookiekim/SOLACE.git
cd SOLACE
pip install -e .
```

**Note:** `flash-attn` is recommended but not auto-installed. Install separately:
```bash
pip install flash-attn --no-build-isolation
```

## Training

### SD3.5-Medium (8 GPUs)
```bash
bash scripts/single_node/grpo_self.sh
# or manually:
accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml \
    --num_processes=8 --main_process_port 29501 \
    scripts/train_sd3_self.py --config config/solace.py:general_ocr_sd3_8gpu
```

### SD3.5-Medium Hybrid (self + external reward)
```bash
accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml \
    --num_processes=8 --main_process_port 29501 \
    scripts/train_sd3_self_ext.py --config config/solace.py:general_ocr_sd3_8gpu
```

### Flux.1-dev (8 GPUs)
```bash
bash scripts/single_node/grpo_flux_self.sh
```

### SDXL (8 GPUs)
```bash
bash scripts/single_node/grpo_self_sdxl.sh
# or specify GPU count:
bash scripts/single_node/grpo_self_sdxl.sh 4 sdxl_self_4gpu
```

### WAN 2.1 Video (8 GPUs)
```bash
bash scripts/single_node/grpo_wan_self.sh
```

## Reward Models

When using **hybrid** training (`train_sd3_self_ext.py`) or external reward evaluation, configure the reward function in the config:

```python
config.reward_fn = {
    "ocr": 1.0,        # OCR accuracy reward
    # "pickscore": 1.0, # PickScore reward
    # "geneval": 1.0,   # GenEval reward
}
```

External reward models are loaded automatically. The OCR reward uses EasyOCR, while PickScore and ImageReward require their respective packages.

## Dataset

Training prompts are provided in `dataset/`. Each subdirectory contains `train.txt` (training prompts) and `test.txt` (evaluation prompts). Specify the dataset path in the config:

```python
config.dataset = os.path.join(os.getcwd(), "dataset/ocr")
```

## Configuration

All configs are in `config/solace.py`. Key parameters:

| Parameter | Description |
|-----------|-------------|
| `config.train.beta` | KL divergence loss weight |
| `config.sample.num_image_per_prompt` | Group size for GRPO |
| `config.sample.global_std` | Use global std for advantage normalization |
| `config.train.ema` | Enable EMA for reference model |
| `config.train.sds.k` | Number of antithetic probes (SDXL) |
| `config.sample.noise_level` | Noise injection level (Flux) |

## Citation

```bibtex
@inproceedings{kim2026solace,
    title={Improving Text-to-Image Generation with Intrinsic Self-Confidence Rewards},
    author={Kim, Wookyoung and others},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2026}
}
```

## Acknowledgments

This codebase is built upon [Flow-GRPO](https://github.com/yifan123/flow_grpo) by Jie Liu et al. We thank the authors for their beautiful open-source framework for applying GRPO to flow-matching diffusion models.

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
