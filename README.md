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

## Pretrained Weights

We release the SOLACE LoRA adapters on the [HuggingFace Hub](https://huggingface.co/wookiekim). Each is a PEFT LoRA adapter for the SD3 transformer — load it on top of the corresponding base model.

| Model | Base | Description | Weights |
|-------|------|-------------|---------|
| SOLACE (SD3.5-M) | SD3.5-Medium | Self-confidence reward only | [`wookiekim/SD3.5M-SOLACE`](https://huggingface.co/wookiekim/SD3.5M-SOLACE) |
| SOLACE (SD3.5-L) | SD3.5-Large | Self-confidence reward only | [`wookiekim/SD3.5L-SOLACE`](https://huggingface.co/wookiekim/SD3.5L-SOLACE) |
| SOLACE (Flux.1-dev) | Flux.1-dev | Self-confidence reward only | [`wookiekim/FLUX.1-dev-SOLACE`](https://huggingface.co/wookiekim/FLUX.1-dev-SOLACE) |
| SOLACE (Wan2.1) | Wan2.1-T2V-1.3B | Self-confidence reward only (text-to-video) | [`wookiekim/Wan2.1-T2V-1.3B-SOLACE`](https://huggingface.co/wookiekim/Wan2.1-T2V-1.3B-SOLACE) |
| SOLACE on FlowGRPO-GenEval | SD3.5-Medium | Flow-GRPO post-trained on GenEval, then SOLACE | [`wookiekim/SD3.5M-SOLACE-on-FlowGRPO-GenEval`](https://huggingface.co/wookiekim/SD3.5M-SOLACE-on-FlowGRPO-GenEval) |
| SOLACE on FlowGRPO-OCR | SD3.5-Medium | Flow-GRPO post-trained on OCR, then SOLACE | [`wookiekim/SD3.5M-SOLACE-on-FlowGRPO-OCR`](https://huggingface.co/wookiekim/SD3.5M-SOLACE-on-FlowGRPO-OCR) |
| SOLACE on FlowGRPO-PickScore | SD3.5-Medium | Flow-GRPO post-trained on PickScore, then SOLACE | [`wookiekim/SD3.5M-SOLACE-on-FlowGRPO-PickScore`](https://huggingface.co/wookiekim/SD3.5M-SOLACE-on-FlowGRPO-PickScore) |

### Inference

```python
import torch
from diffusers import StableDiffusion3Pipeline
from peft import PeftModel

model_id = "stabilityai/stable-diffusion-3.5-medium"  # or -large for SD3.5L
lora_ckpt_path = "wookiekim/SD3.5M-SOLACE-on-FlowGRPO-GenEval"
device = "cuda"

pipe = StableDiffusion3Pipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.transformer = PeftModel.from_pretrained(pipe.transformer, lora_ckpt_path)
pipe.transformer = pipe.transformer.merge_and_unload()
pipe = pipe.to(device)

image = pipe(
    "a photo of a black kite and a green bear",
    height=512, width=512, num_inference_steps=40, guidance_scale=4.5,
).images[0]
image.save("solace.png")
```

To continue training from a released adapter, point `config.train.lora_path` at the HuggingFace repo id (or a local download).

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
