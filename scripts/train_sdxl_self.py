"""
SDXL GRPO + Self-Certainty Training Script
===========================================
Mirrors train_sd3_self.py but adapted for SDXL (UNet, epsilon-prediction,
DDPMScheduler for stochastic logprob).

Expected GPU memory (LoRA, bf16, 1024x1024):
  - ~24 GB with batch_size=1, group_size=2
  - ~40 GB with batch_size=2, group_size=4

Minimal debug command:
  accelerate launch --num_processes=1 scripts/train_sdxl_self.py \
      --config config/solace.py:sdxl_self_8gpu --debug_sanity

Main pitfalls:
  - SDXL requires 2 text encoders + pooled embeddings + add_time_ids
  - DDPMScheduler must be used (not DDIM/Euler) for valid stochastic logprob
  - Epsilon prediction: noise_pred is eps, not velocity
  - LoRA targets UNet attention (to_q, to_k, to_v, to_out.0), not transformer
"""

from collections import defaultdict
import contextlib
import os
import datetime
from concurrent import futures
import time
import json
import hashlib
import math
from absl import app, flags
from accelerate import Accelerator
from ml_collections import config_flags
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from diffusers import StableDiffusionXLPipeline, DDPMScheduler, UNet2DConditionModel, AutoencoderKL
from diffusers.utils.torch_utils import is_compiled_module, randn_tensor
import numpy as np
import solace.prompts
import solace.rewards
from solace.stat_tracking import PerPromptStatTracker
import torch
import wandb
from functools import partial
import tqdm
import tempfile
from PIL import Image
from peft import LoraConfig, get_peft_model, PeftModel
import random
from torch.utils.data import Dataset, DataLoader, Sampler
from solace.ema import EMAModuleWrapper

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")
flags.DEFINE_boolean("debug_sanity", False, "Run 1-batch sanity check and exit.")
logger = get_logger(__name__)


def adapter_off_ctx(m):
    """Returns a context manager that disables LoRA adapters if present."""
    base = getattr(m, "module", m)
    if hasattr(base, "disable_adapter"):
        return base.disable_adapter()
    return contextlib.nullcontext()


# --------------------- Datasets & Sampler (identical to SD3) ---------------------

class TextPromptDataset(Dataset):
    def __init__(self, dataset, split='train'):
        self.file_path = os.path.join(dataset, f'{split}.txt')
        with open(self.file_path, 'r') as f:
            self.prompts = [line.strip() for line in f.readlines()]
    def __len__(self):
        return len(self.prompts)
    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": {}}
    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas


class GenevalPromptDataset(Dataset):
    def __init__(self, dataset, split='train'):
        self.file_path = os.path.join(dataset, f'{split}_metadata.jsonl')
        with open(self.file_path, 'r', encoding='utf-8') as f:
            self.metadatas = [json.loads(line) for line in f]
            self.prompts = [item['prompt'] for item in self.metadatas]
    def __len__(self):
        return len(self.prompts)
    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": self.metadatas[idx]}
    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas


class DistributedKRepeatSampler(Sampler):
    def __init__(self, dataset, batch_size, k, num_replicas, rank, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.k = k
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.total_samples = self.num_replicas * self.batch_size
        assert self.total_samples % self.k == 0
        self.m = self.total_samples // self.k
        self.epoch = 0
    def __iter__(self):
        while True:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g)[:self.m].tolist()
            repeated_indices = [idx for idx in indices for _ in range(self.k)]
            shuffled_indices = torch.randperm(len(repeated_indices), generator=g).tolist()
            shuffled_samples = [repeated_indices[i] for i in shuffled_indices]
            per_card_samples = []
            for i in range(self.num_replicas):
                start = i * self.batch_size
                end = start + self.batch_size
                per_card_samples.append(shuffled_samples[start:end])
            yield per_card_samples[self.rank]
    def set_epoch(self, epoch):
        self.epoch = epoch


# --------------------- SDXL Text Encoding ---------------------

def encode_prompt_sdxl(tokenizers, text_encoders, prompts, device):
    """Encode prompts for SDXL (2 text encoders, returns prompt_embeds + pooled)."""
    # Tokenizer 1 (CLIP ViT-L)
    text_input_1 = tokenizers[0](
        prompts, padding="max_length", max_length=tokenizers[0].model_max_length,
        truncation=True, return_tensors="pt",
    )
    text_input_ids_1 = text_input_1.input_ids.to(device)

    # Tokenizer 2 (OpenCLIP ViT-bigG)
    text_input_2 = tokenizers[1](
        prompts, padding="max_length", max_length=tokenizers[1].model_max_length,
        truncation=True, return_tensors="pt",
    )
    text_input_ids_2 = text_input_2.input_ids.to(device)

    with torch.no_grad():
        encoder_output_1 = text_encoders[0](text_input_ids_1, output_hidden_states=True)
        prompt_embeds_1 = encoder_output_1.hidden_states[-2]  # penultimate layer

        encoder_output_2 = text_encoders[1](text_input_ids_2, output_hidden_states=True)
        prompt_embeds_2 = encoder_output_2.hidden_states[-2]  # penultimate layer
        pooled_prompt_embeds = encoder_output_2[0]  # pooled output from text_encoder_2

    prompt_embeds = torch.cat([prompt_embeds_1, prompt_embeds_2], dim=-1)
    return prompt_embeds.to(device), pooled_prompt_embeds.to(device)


def compute_sdxl_add_time_ids(original_size, crops_coords_top_left, target_size, dtype, device):
    """Compute add_time_ids for SDXL conditioning."""
    add_time_ids = list(original_size + crops_coords_top_left + target_size)
    add_time_ids = torch.tensor([add_time_ids], dtype=dtype, device=device)
    return add_time_ids


# --------------------- Helpers (from SD3) ---------------------

def calculate_zero_std_ratio(prompts, gathered_rewards):
    prompt_array = np.array(prompts)
    unique_prompts, inverse_indices, counts = np.unique(
        prompt_array, return_inverse=True, return_counts=True
    )
    grouped_rewards = gathered_rewards['ori_avg'][np.argsort(inverse_indices)]
    split_indices = np.cumsum(counts)[:-1]
    reward_groups = np.split(grouped_rewards, split_indices)
    prompt_std_devs = np.array([np.std(group) for group in reward_groups])
    zero_std_count = np.count_nonzero(prompt_std_devs == 0)
    zero_std_ratio = zero_std_count / len(prompt_std_devs)
    return zero_std_ratio, prompt_std_devs.mean()


def create_generator(prompts, base_seed):
    generators = []
    for prompt in prompts:
        hash_digest = hashlib.sha256(prompt.encode()).digest()
        prompt_hash_int = int.from_bytes(hash_digest[:4], 'big')
        seed = (base_seed + prompt_hash_int) % (2**31)
        gen = torch.Generator().manual_seed(seed)
        generators.append(gen)
    return generators


def unwrap_model(model, accelerator):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model


def save_ckpt(save_dir, unet, global_step, accelerator, ema, trainable_parameters, config):
    save_root = os.path.join(save_dir, "checkpoints", f"checkpoint-{global_step}")
    save_root_lora = os.path.join(save_root, "lora")
    os.makedirs(save_root_lora, exist_ok=True)
    if accelerator.is_main_process:
        if config.train.ema:
            ema.copy_ema_to(trainable_parameters, store_temp=True)
        unwrap_model(unet, accelerator).save_pretrained(save_root_lora)
        if config.train.ema:
            ema.copy_temp_to(trainable_parameters)


# --------------------- DDPM Stochastic Sampling with LogProb ---------------------

def ddpm_step_with_logprob(
    scheduler: DDPMScheduler,
    noise_pred: torch.FloatTensor,
    timestep: torch.LongTensor,
    sample: torch.FloatTensor,
    prev_sample: torch.FloatTensor = None,
    generator=None,
):
    """
    Single DDPM reverse step with Gaussian log-probability computation.
    For epsilon-prediction: x_{t-1} ~ N(mu_theta(x_t, t), sigma_t^2 I)
    where mu_theta = (1/sqrt(alpha_t)) * (x_t - beta_t/sqrt(1-alpha_bar_t) * eps_theta)
    and sigma_t^2 = posterior variance (beta_tilde_t).

    IMPORTANT: prev_t must be computed using the scheduler's step spacing,
    NOT t-1, because set_timesteps() selects a subset of the full schedule.

    Returns: (prev_sample, log_prob, prev_sample_mean, std_dev_t)
    """
    # Ensure float32 for numerical stability
    noise_pred = noise_pred.float()
    sample = sample.float()
    if prev_sample is not None:
        prev_sample = prev_sample.float()

    t = timestep
    # Compute previous timestep using scheduler's step spacing (matches diffusers)
    num_train_timesteps = scheduler.config.num_train_timesteps
    step_ratio = num_train_timesteps // scheduler.num_inference_steps

    # Handle both scalar and batch timesteps
    if t.dim() == 0:
        t_idx = t.item()
        prev_t_idx = max(t_idx - step_ratio, 0)
        alpha_prod_t = scheduler.alphas_cumprod[t_idx]
        alpha_prod_t_prev = scheduler.alphas_cumprod[prev_t_idx] if prev_t_idx > 0 else torch.tensor(1.0)
        beta_prod_t = 1.0 - alpha_prod_t
        beta_prod_t_prev = 1.0 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1.0 - current_alpha_t
    else:
        # Batch of timesteps — index on CPU, then move to device
        t_cpu = t.cpu()
        prev_t_cpu = torch.clamp(t_cpu - step_ratio, min=0)
        alpha_prod_t = scheduler.alphas_cumprod[t_cpu].view(-1, 1, 1, 1).to(sample.device)
        # For prev_t == 0, use alpha_prod = 1.0 (no noise at t=0)
        alpha_prod_t_prev = torch.where(
            prev_t_cpu > 0,
            scheduler.alphas_cumprod[prev_t_cpu],
            torch.ones_like(scheduler.alphas_cumprod[prev_t_cpu]),
        ).view(-1, 1, 1, 1).to(sample.device)
        beta_prod_t = 1.0 - alpha_prod_t
        beta_prod_t_prev = 1.0 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1.0 - current_alpha_t

    # Predicted x_0
    pred_original_sample = (sample - torch.sqrt(beta_prod_t) * noise_pred) / torch.sqrt(alpha_prod_t)

    # Clip predicted x_0
    if scheduler.config.clip_sample:
        pred_original_sample = pred_original_sample.clamp(
            -scheduler.config.clip_sample_range, scheduler.config.clip_sample_range
        )

    # Compute posterior mean: mu_theta
    pred_original_sample_coeff = torch.sqrt(alpha_prod_t_prev) * current_beta_t / beta_prod_t
    current_sample_coeff = torch.sqrt(current_alpha_t) * beta_prod_t_prev / beta_prod_t
    prev_sample_mean = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample

    # Posterior variance
    variance = current_beta_t * beta_prod_t_prev / beta_prod_t
    # Clamp for numerical stability
    variance = torch.clamp(variance, min=1e-20)
    std_dev_t = torch.sqrt(variance)

    if prev_sample is None:
        noise = randn_tensor(
            noise_pred.shape, generator=generator,
            device=noise_pred.device, dtype=noise_pred.dtype,
        )
        prev_sample = prev_sample_mean + std_dev_t * noise

    # Log probability: log N(x_{t-1} | mu, sigma^2)
    log_prob = (
        -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * variance)
        - torch.log(std_dev_t)
        - 0.5 * math.log(2 * math.pi)
    )
    # Mean over all spatial dimensions
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

    return prev_sample, log_prob, prev_sample_mean, std_dev_t


# --------------------- SDXL Sampling with LogProb ---------------------

@torch.no_grad()
def sdxl_sample_with_logprob(
    unet,
    scheduler: DDPMScheduler,
    prompt_embeds,
    pooled_prompt_embeds,
    add_time_ids,
    neg_prompt_embeds=None,
    neg_pooled_prompt_embeds=None,
    neg_add_time_ids=None,
    num_inference_steps=20,
    guidance_scale=7.5,
    height=1024,
    width=1024,
    latent_channels=4,
    vae_scale_factor=8,
    generator=None,
    device=None,
    dtype=None,
    logprob_step_indices=None,
):
    """
    SDXL sampling from x_T to x_0 using DDPMScheduler with per-step logprob.

    Args:
        logprob_step_indices: list of step indices for which to compute logprob.
            If None, compute for all steps.
    Returns:
        (all_latents, all_log_probs, timesteps_tensor)
        all_latents: list of latents [x_T, x_{T-1}, ..., x_0], length = num_steps + 1
        all_log_probs: list of log_probs per step, length = num_steps
    """
    batch_size = prompt_embeds.shape[0]
    latent_h = height // vae_scale_factor
    latent_w = width // vae_scale_factor

    # Initial noise
    latents = randn_tensor(
        (batch_size, latent_channels, latent_h, latent_w),
        generator=generator, device=device, dtype=torch.float32,
    )

    # Set timesteps
    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps

    do_cfg = guidance_scale > 1.0 and neg_prompt_embeds is not None

    # Expand add_time_ids for batch
    add_time_ids_batch = add_time_ids.repeat(batch_size, 1)
    if do_cfg:
        neg_add_time_ids_batch = neg_add_time_ids.repeat(batch_size, 1) if neg_add_time_ids is not None else add_time_ids_batch

    all_latents = [latents]
    all_log_probs = []

    # Determine which steps to compute logprob for
    if logprob_step_indices is not None:
        logprob_set = set(logprob_step_indices)
    else:
        logprob_set = set(range(len(timesteps)))

    # Determine UNet weight dtype for casting inputs
    unet_dtype = next(unet.parameters()).dtype
    # Cast all conditioning to UNet dtype for consistency
    prompt_embeds = prompt_embeds.to(unet_dtype)
    pooled_prompt_embeds = pooled_prompt_embeds.to(unet_dtype)
    add_time_ids_batch = add_time_ids_batch.to(unet_dtype)
    if do_cfg:
        neg_prompt_embeds = neg_prompt_embeds.to(unet_dtype)
        neg_pooled_prompt_embeds = neg_pooled_prompt_embeds.to(unet_dtype)
        neg_add_time_ids_batch = neg_add_time_ids_batch.to(unet_dtype)

    for i, t in enumerate(timesteps):
        latent_input = latents.to(unet_dtype)
        t_batch = t.expand(batch_size)

        if do_cfg:
            latent_input_cfg = torch.cat([latent_input, latent_input])
            t_cfg = torch.cat([t_batch, t_batch])
            embeds_cfg = torch.cat([neg_prompt_embeds, prompt_embeds])
            pooled_cfg = torch.cat([neg_pooled_prompt_embeds, pooled_prompt_embeds])
            time_ids_cfg = torch.cat([neg_add_time_ids_batch, add_time_ids_batch])

            noise_pred = unet(
                latent_input_cfg,
                t_cfg,
                encoder_hidden_states=embeds_cfg,
                added_cond_kwargs={"text_embeds": pooled_cfg, "time_ids": time_ids_cfg},
                return_dict=False,
            )[0]
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        else:
            noise_pred = unet(
                latent_input,
                t_batch,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs={
                    "text_embeds": pooled_prompt_embeds,
                    "time_ids": add_time_ids_batch,
                },
                return_dict=False,
            )[0]

        # DDPM step with logprob (always in float32 for numerical stability)
        latents, log_prob, _, _ = ddpm_step_with_logprob(
            scheduler, noise_pred.float(), t, latents.float(),
        )

        all_latents.append(latents)
        if i in logprob_set:
            all_log_probs.append(log_prob)
        else:
            all_log_probs.append(torch.zeros(batch_size, device=device))

    return all_latents, all_log_probs, timesteps


# --------------------- LogProb Recomputation for Training ---------------------

def compute_log_prob_sdxl(
    unet, scheduler, sample, j, prompt_embeds, pooled_prompt_embeds,
    add_time_ids, neg_prompt_embeds, neg_pooled_prompt_embeds,
    neg_add_time_ids, config,
):
    """Recompute logprob at step j for GRPO training."""
    batch_size = sample["latents"].shape[0]
    add_time_ids_batch = add_time_ids.repeat(batch_size, 1)
    t = sample["timesteps"][:, j]  # [B]
    unet_dtype = next(unet.parameters()).dtype

    if config.train.cfg:
        neg_add_time_ids_batch = neg_add_time_ids.repeat(batch_size, 1) if neg_add_time_ids is not None else add_time_ids_batch
        latent_input = torch.cat([sample["latents"][:, j]] * 2).to(unet_dtype)
        t_input = torch.cat([t, t])
        embeds = torch.cat([neg_prompt_embeds[:batch_size], prompt_embeds]).to(unet_dtype)
        pooled = torch.cat([neg_pooled_prompt_embeds[:batch_size], pooled_prompt_embeds]).to(unet_dtype)
        time_ids = torch.cat([neg_add_time_ids_batch, add_time_ids_batch]).to(unet_dtype)

        noise_pred = unet(
            latent_input,
            t_input,
            encoder_hidden_states=embeds,
            added_cond_kwargs={"text_embeds": pooled, "time_ids": time_ids},
            return_dict=False,
        )[0]
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + config.sample.guidance_scale * (noise_pred_text - noise_pred_uncond)
    else:
        noise_pred = unet(
            sample["latents"][:, j].to(unet_dtype),
            t,
            encoder_hidden_states=prompt_embeds.to(unet_dtype),
            added_cond_kwargs={
                "text_embeds": pooled_prompt_embeds.to(unet_dtype),
                "time_ids": add_time_ids_batch.to(unet_dtype),
            },
            return_dict=False,
        )[0]

    prev_sample, log_prob, prev_sample_mean, std_dev_t = ddpm_step_with_logprob(
        scheduler, noise_pred.float(), t, sample["latents"][:, j].float(),
        prev_sample=sample["next_latents"][:, j].float(),
    )
    return prev_sample, log_prob, prev_sample_mean, std_dev_t


# --------------------- Self-Certainty Reward (SDXL: epsilon prediction) ----------

def sdxl_self_certainty_reward(
    unet,
    x0,
    scheduler: DDPMScheduler,
    prompt_embeds,
    pooled_prompt_embeds,
    add_time_ids,
    neg_prompt_embeds,
    neg_pooled_prompt_embeds,
    neg_add_time_ids,
    config,
    device,
    autocast_ctx,
    num_reward_timesteps=8,
):
    """
    Self-certainty reward for SDXL in latent space (epsilon prediction).

    For each t in a set of reward timesteps:
      z_t = sqrt(alpha_bar_t) * z0 + sqrt(1 - alpha_bar_t) * eps
      eps_hat = unet(z_t, t, cond)
      mse = mean((eps_hat - eps)^2)
      reward = -mean_t(mse)

    Uses K antithetic probes per timestep for variance reduction.
    """
    B, C, H, W = x0.shape
    K = getattr(config.train, "sds", {}).get("k", 8)
    step_stride = getattr(config.train, "sds", {}).get("use_step_stride", 1)
    scale = getattr(config.train, "sds", {}).get("scale", 1.0)

    # Choose reward timesteps: evenly spaced in [0, T-1]
    T = scheduler.config.num_train_timesteps
    if num_reward_timesteps >= T:
        reward_ts = list(range(T))
    else:
        reward_ts = [int(T * (i + 0.5) / num_reward_timesteps) for i in range(num_reward_timesteps)]

    reward_ts_tensor = torch.tensor(reward_ts, device=device, dtype=torch.long)

    # Determine UNet weight dtype for casting inputs
    unet_dtype = next(unet.parameters()).dtype

    # Expand embeddings for K probes, cast to UNet dtype
    cond_pe = prompt_embeds.repeat(K, 1, 1).to(unet_dtype)
    cond_pp = pooled_prompt_embeds.repeat(K, 1).to(unet_dtype)
    add_ids = add_time_ids.repeat(K * B, 1).to(unet_dtype)

    sds_per_step = torch.zeros(B, len(reward_ts), device=device, dtype=torch.float32)

    with torch.no_grad():
        with autocast_ctx():
            for idx, t_val in enumerate(reward_ts):
                if idx % step_stride != 0:
                    continue
                t_tensor = torch.full((K * B,), t_val, device=device, dtype=torch.long)

                alpha_bar_t = scheduler.alphas_cumprod[t_val].to(device)
                sqrt_alpha = torch.sqrt(alpha_bar_t)
                sqrt_one_minus_alpha = torch.sqrt(1.0 - alpha_bar_t)

                # K antithetic probes
                eps = torch.randn(K // 2, B, C, H, W, device=device, dtype=x0.dtype)
                eps = torch.cat((eps, -eps), dim=0)  # [K, B, C, H, W]

                x0_rep = x0.unsqueeze(0).expand(K, -1, -1, -1, -1)  # [K,B,C,H,W]
                zt = sqrt_alpha * x0_rep + sqrt_one_minus_alpha * eps  # [K,B,C,H,W]
                zt_flat = zt.reshape(K * B, C, H, W).to(unet_dtype)

                eps_hat = unet(
                    zt_flat,
                    t_tensor,
                    encoder_hidden_states=cond_pe,
                    added_cond_kwargs={"text_embeds": cond_pp, "time_ids": add_ids},
                    return_dict=False,
                )[0]

                eps_flat = eps.reshape(K * B, C, H, W)
                mse_flat = torch.mean((eps_hat.float() - eps_flat.float()) ** 2, dim=(1, 2, 3))
                mse = mse_flat.view(K, B).mean(dim=0)
                sds_step = -torch.log(mse + 1e-6)
                sds_per_step[:, idx] = sds_step

    # Normalize across batch per step (handle B=1 gracefully)
    mean_t = sds_per_step.mean(dim=0, keepdim=True)
    if B > 1:
        std_t = sds_per_step.std(dim=0, keepdim=True).clamp_min(1e-6)
        sds_norm = (sds_per_step - mean_t) / std_t
    else:
        sds_norm = sds_per_step - mean_t  # zero-centered, no std normalization

    # Time weighting: emphasize mid-range timesteps
    t_frac = reward_ts_tensor.float() / T
    weight = t_frac * (1.0 - t_frac)
    sds_norm = sds_norm * weight.unsqueeze(0)

    sds_scalar = scale * sds_norm.mean(dim=1)
    return sds_scalar, sds_norm, sds_per_step


# --------------------- Main ---------------------

def main(_):
    config = FLAGS.config
    debug_sanity = FLAGS.debug_sanity

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id

    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)

    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps * max(1, num_train_timesteps),
    )
    if accelerator.is_main_process:
        wandb.init(project="solace")
    logger.info(f"\n{config}")

    set_seed(config.seed, device_specific=True)

    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    # ---- Load SDXL Pipeline ----
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        config.pretrained.model, torch_dtype=inference_dtype,
    )

    # Freeze everything except UNet (or LoRA)
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    pipeline.unet.requires_grad_(not config.use_lora)

    text_encoders = [pipeline.text_encoder, pipeline.text_encoder_2]
    tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2]
    pipeline.safety_checker = None

    pipeline.vae.to(accelerator.device, dtype=torch.float32)
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder_2.to(accelerator.device, dtype=inference_dtype)
    pipeline.unet.to(accelerator.device)

    # ---- Replace scheduler with DDPMScheduler for stochastic logprob ----
    ddpm_scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)
    # Keep pipeline scheduler for reference, but use ddpm_scheduler for training
    pipeline.scheduler = ddpm_scheduler

    # ---- Enable optimizations ----
    if hasattr(pipeline.unet, "enable_gradient_checkpointing"):
        pipeline.unet.enable_gradient_checkpointing()
    try:
        pipeline.unet.enable_xformers_memory_efficient_attention()
    except Exception:
        logger.info("xformers not available, using default attention")

    # ---- LoRA ----
    if config.use_lora:
        sdxl_lora_target_modules = getattr(config, "sdxl_lora_target_modules", None)
        if sdxl_lora_target_modules is None:
            sdxl_lora_target_modules = [
                "to_q", "to_k", "to_v", "to_out.0",
            ]
        lora_rank = getattr(config, "lora_rank", 32)
        lora_alpha = getattr(config, "lora_alpha", 64)
        lora_dropout = getattr(config, "lora_dropout", 0.0)

        unet_lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights="gaussian",
            target_modules=sdxl_lora_target_modules,
        )
        lora_path = getattr(config.train, "lora_path", None)
        if lora_path:
            pipeline.unet = PeftModel.from_pretrained(pipeline.unet, lora_path)
            pipeline.unet.set_adapter("default")
        else:
            pipeline.unet = get_peft_model(pipeline.unet, unet_lora_config)

    unet = pipeline.unet
    unet_trainable_parameters = list(filter(lambda p: p.requires_grad, unet.parameters()))
    ema = EMAModuleWrapper(unet_trainable_parameters, decay=0.9, update_step_interval=8, device=accelerator.device)

    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if config.train.use_8bit_adam:
        import bitsandbytes as bnb
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        unet_trainable_parameters,
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    # External reward functions (for evaluation) — skip in debug mode to save GPU memory
    if not debug_sanity:
        reward_fn = getattr(solace.rewards, 'multi_score')(accelerator.device, config.reward_fn)
        eval_reward_fn = getattr(solace.rewards, 'multi_score')(accelerator.device, config.reward_fn)
        executor = futures.ThreadPoolExecutor(max_workers=8)

    # ---- Datasets ----
    if config.prompt_fn == "general_ocr":
        train_dataset = TextPromptDataset(config.dataset, 'train')
        test_dataset = TextPromptDataset(config.dataset, 'test')
        train_sampler = DistributedKRepeatSampler(
            train_dataset, config.sample.train_batch_size,
            config.sample.num_image_per_prompt, accelerator.num_processes,
            accelerator.process_index, seed=42,
        )
        train_dataloader = DataLoader(train_dataset, batch_sampler=train_sampler,
                                       num_workers=1, collate_fn=TextPromptDataset.collate_fn)
        test_dataloader = DataLoader(test_dataset, batch_size=config.sample.test_batch_size,
                                      collate_fn=TextPromptDataset.collate_fn, shuffle=False, num_workers=8)
    elif config.prompt_fn == "geneval":
        train_dataset = GenevalPromptDataset(config.dataset, 'train')
        test_dataset = GenevalPromptDataset(config.dataset, 'test')
        train_sampler = DistributedKRepeatSampler(
            train_dataset, config.sample.train_batch_size,
            config.sample.num_image_per_prompt, accelerator.num_processes,
            accelerator.process_index, seed=42,
        )
        train_dataloader = DataLoader(train_dataset, batch_sampler=train_sampler,
                                       num_workers=1, collate_fn=GenevalPromptDataset.collate_fn)
        test_dataloader = DataLoader(test_dataset, batch_size=config.sample.test_batch_size,
                                      collate_fn=GenevalPromptDataset.collate_fn, shuffle=False, num_workers=8)
    else:
        raise NotImplementedError("Only general_ocr and geneval are supported.")

    # ---- SDXL conditioning setup ----
    resolution = config.resolution
    original_size = (resolution, resolution)
    target_size = (resolution, resolution)
    crops_coords_top_left = (0, 0)
    add_time_ids = compute_sdxl_add_time_ids(
        original_size, crops_coords_top_left, target_size,
        dtype=inference_dtype, device=accelerator.device,
    )
    neg_add_time_ids = add_time_ids.clone()

    # Negative prompt embeddings
    neg_prompt_embeds, neg_pooled_prompt_embeds = encode_prompt_sdxl(
        tokenizers, text_encoders, [""], accelerator.device,
    )
    sample_neg_prompt_embeds = neg_prompt_embeds.repeat(config.sample.train_batch_size, 1, 1)
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embeds.repeat(config.sample.train_batch_size, 1)
    train_neg_prompt_embeds = neg_prompt_embeds.repeat(config.train.batch_size, 1, 1)
    train_neg_pooled_prompt_embeds = neg_pooled_prompt_embeds.repeat(config.train.batch_size, 1)

    if config.sample.num_image_per_prompt == 1:
        config.per_prompt_stat_tracking = False
    if config.per_prompt_stat_tracking:
        stat_tracker = PerPromptStatTracker(config.sample.global_std)

    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast

    # Logprob step indices (configurable subset)
    logprob_steps = getattr(config.sample, "logprob_steps", None)
    if logprob_steps is not None and logprob_steps < config.sample.num_steps:
        logprob_step_indices = [
            int(config.sample.num_steps * (i + 0.5) / logprob_steps)
            for i in range(logprob_steps)
        ]
        logger.info(f"Computing logprob on {logprob_steps}/{config.sample.num_steps} steps: {logprob_step_indices}")
    else:
        logprob_step_indices = None

    unet, optimizer, train_dataloader, test_dataloader = accelerator.prepare(
        unet, optimizer, train_dataloader, test_dataloader,
    )

    # Reward timestep count
    num_reward_timesteps = getattr(config.train, "num_reward_timesteps", 8)

    samples_per_epoch = config.sample.train_batch_size * accelerator.num_processes * config.sample.num_batches_per_epoch
    total_train_batch_size = config.train.batch_size * accelerator.num_processes * config.train.gradient_accumulation_steps

    logger.info("***** Running SDXL GRPO + Self-Certainty Training *****")
    logger.info(f"  Sample batch size per device = {config.sample.train_batch_size}")
    logger.info(f"  Train batch size per device  = {config.train.batch_size}")
    logger.info(f"  Grad Accum steps (x timesteps) = {config.train.gradient_accumulation_steps} x {num_train_timesteps}")
    logger.info(f"  Total samples/epoch = {samples_per_epoch}")
    logger.info(f"  Inner epochs = {config.train.num_inner_epochs}")
    logger.info(f"  Resolution = {resolution}x{resolution}")
    logger.info(f"  Logprob steps = {logprob_steps if logprob_steps else 'all'}")

    epoch = 0
    global_step = 0
    train_iter = iter(train_dataloader)

    # ----- Debug sanity check -----
    if debug_sanity:
        import sys
        print("=== DEBUG SANITY CHECK ===", flush=True)
        prompts_dbg = ["a photo of a cat"]
        pe, ppe = encode_prompt_sdxl(tokenizers, text_encoders, prompts_dbg, accelerator.device)
        print(f"  prompt_embeds shape: {pe.shape}, pooled shape: {ppe.shape}", flush=True)

        print("  Sampling...", flush=True)
        with torch.no_grad():
            all_lat, all_lp, ts = sdxl_sample_with_logprob(
                pipeline.unet, ddpm_scheduler, pe, ppe, add_time_ids,
                neg_prompt_embeds=neg_prompt_embeds,
                neg_pooled_prompt_embeds=neg_pooled_prompt_embeds,
                neg_add_time_ids=neg_add_time_ids,
                num_inference_steps=config.sample.num_steps,
                guidance_scale=config.sample.guidance_scale,
                height=resolution, width=resolution,
                device=accelerator.device, dtype=inference_dtype,
            )
        z0 = all_lat[-1]
        print(f"  sampled z0 mean: {z0.mean().item():.4f}, std: {z0.std().item():.4f}", flush=True)
        logp = torch.stack(all_lp).sum(dim=0)
        print(f"  logp_rel mean: {logp.mean().item():.4f}, std: {logp.std().item():.4f}", flush=True)

        print("  Computing reward...", flush=True)
        reward, _, _ = sdxl_self_certainty_reward(
            pipeline.unet, z0, ddpm_scheduler,
            pe, ppe, add_time_ids,
            neg_prompt_embeds, neg_pooled_prompt_embeds, neg_add_time_ids,
            config, accelerator.device, autocast,
            num_reward_timesteps=num_reward_timesteps,
        )
        print(f"  reward mean: {reward.mean().item():.4f}, std: {reward.std().item():.4f}", flush=True)

        # Check grad flow into LoRA
        if config.use_lora:
            print("  Checking LoRA gradient flow...", flush=True)
            unet.train()
            unet_dtype = next(unet.parameters()).dtype
            dummy_lat = torch.randn(1, 4, resolution // 8, resolution // 8, device=accelerator.device, dtype=unet_dtype)
            dummy_t = torch.tensor([500], device=accelerator.device)
            out = unet(
                dummy_lat, dummy_t, encoder_hidden_states=pe.to(unet_dtype),
                added_cond_kwargs={"text_embeds": ppe.to(unet_dtype), "time_ids": add_time_ids},
            ).sample
            loss = out.mean()
            loss.backward()
            for name, param in unet.named_parameters():
                if param.requires_grad and param.grad is not None:
                    print(f"  LoRA grad norm ({name}): {param.grad.norm().item():.6f}", flush=True)
                    break
            optimizer.zero_grad()

        print("=== SANITY CHECK PASSED ===", flush=True)
        return

    # ----- Main training loop -----
    while True:
        # ========== EVAL ==========
        pipeline.unet.eval()
        if epoch % config.eval_freq == 0:
            _eval_sdxl(
                pipeline, ddpm_scheduler, test_dataloader, tokenizers, text_encoders,
                config, accelerator, global_step, eval_reward_fn, executor,
                autocast, add_time_ids, neg_add_time_ids,
                neg_prompt_embeds, neg_pooled_prompt_embeds,
                num_reward_timesteps, ema, unet_trainable_parameters,
            )
        if epoch % config.save_freq == 0 and epoch > 0 and accelerator.is_main_process:
            save_ckpt(config.save_dir, unet, global_step, accelerator, ema, unet_trainable_parameters, config)

        # ========== SAMPLING + Self-Certainty ==========
        pipeline.unet.eval()
        samples = []
        prompts_last = None
        images_last = None

        for i in tqdm(range(config.sample.num_batches_per_epoch),
                      desc=f"Epoch {epoch}: sampling", disable=not accelerator.is_local_main_process, position=0):
            train_sampler.set_epoch(epoch * config.sample.num_batches_per_epoch + i)
            prompts, prompt_metadata = next(train_iter)

            prompt_embeds, pooled_prompt_embeds = encode_prompt_sdxl(
                tokenizers, text_encoders, prompts, accelerator.device,
            )
            prompt_ids = tokenizers[0](
                prompts, padding="max_length", max_length=256, truncation=True, return_tensors="pt",
            ).input_ids.to(accelerator.device)

            generator = create_generator(prompts, base_seed=epoch * 10000 + i) if config.sample.same_latent else None
            with autocast():
                with torch.no_grad():
                    all_latents, all_log_probs, timesteps_out = sdxl_sample_with_logprob(
                        pipeline.unet, ddpm_scheduler,
                        prompt_embeds, pooled_prompt_embeds, add_time_ids,
                        neg_prompt_embeds=sample_neg_prompt_embeds,
                        neg_pooled_prompt_embeds=sample_neg_pooled_prompt_embeds,
                        neg_add_time_ids=neg_add_time_ids,
                        num_inference_steps=config.sample.num_steps,
                        guidance_scale=config.sample.guidance_scale,
                        height=resolution, width=resolution,
                        device=accelerator.device, dtype=inference_dtype,
                        generator=generator,
                        logprob_step_indices=logprob_step_indices,
                    )

            latents = torch.stack(all_latents, dim=1)       # [B, T+1, C, H, W]
            log_probs = torch.stack(all_log_probs, dim=1)    # [B, T]
            timesteps = timesteps_out.unsqueeze(0).repeat(len(prompts), 1).to(accelerator.device)

            x0 = latents[:, -1]

            # Decode to images for logging (periodically)
            if i == config.sample.num_batches_per_epoch - 1:
                with torch.no_grad():
                    decoded = pipeline.vae.decode(
                        x0.to(pipeline.vae.dtype) / pipeline.vae.config.scaling_factor
                    ).sample
                    decoded = (decoded / 2 + 0.5).clamp(0, 1)
                    images_last = decoded.cpu()
                    prompts_last = prompts

            # Self-certainty reward
            sds_scalar, sds_norm, sds_per_step = sdxl_self_certainty_reward(
                pipeline.unet, x0, ddpm_scheduler,
                prompt_embeds, pooled_prompt_embeds, add_time_ids,
                neg_prompt_embeds, neg_pooled_prompt_embeds, neg_add_time_ids,
                config, accelerator.device, autocast,
                num_reward_timesteps=num_reward_timesteps,
            )

            samples.append({
                "prompt_ids": prompt_ids,
                "prompt_embeds": prompt_embeds,
                "pooled_prompt_embeds": pooled_prompt_embeds,
                "timesteps": timesteps,
                "latents": latents[:, :-1],
                "next_latents": latents[:, 1:],
                "log_probs": log_probs,
                "rewards": {"avg": sds_scalar},
                "sds_per_step": sds_per_step,
            })

        # Collate
        samples = {
            k: torch.cat([s[k] for s in samples], dim=0)
            if not isinstance(samples[0][k], dict)
            else {sub_key: torch.cat([s[k][sub_key] for s in samples], dim=0) for sub_key in samples[0][k]}
            for k in samples[0].keys()
        }

        # Log images
        if epoch % 10 == 0 and accelerator.is_main_process and images_last is not None:
            with tempfile.TemporaryDirectory() as tmpdir:
                num_samples = min(15, len(images_last))
                sample_indices = random.sample(range(len(images_last)), num_samples)
                for idx, ii in enumerate(sample_indices):
                    image = images_last[ii]
                    pil = Image.fromarray((image.numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                    pil = pil.resize((resolution, resolution))
                    pil.save(os.path.join(tmpdir, f"{idx}.jpg"))
                sampled_prompts = [prompts_last[ii] for ii in sample_indices]
                sampled_rewards = [samples["rewards"]["avg"][ii].item() for ii in sample_indices]
                wandb.log({
                    "images": [
                        wandb.Image(os.path.join(tmpdir, f"{idx}.jpg"),
                                    caption=f"{prompt:.100} | reward: {avg_reward:.2f}")
                        for idx, (prompt, avg_reward) in enumerate(zip(sampled_prompts, sampled_rewards))
                    ],
                }, step=global_step)

        samples["rewards"]["ori_avg"] = samples["rewards"]["avg"].clone()
        samples["rewards"]["sds_per_step"] = samples["sds_per_step"].clone()
        samples["rewards"]["avg"] = samples["rewards"]["avg"].unsqueeze(1).repeat(1, num_train_timesteps)

        gathered_rewards = {key: accelerator.gather(value) for key, value in samples["rewards"].items()}
        gathered_rewards = {key: value.cpu().numpy() for key, value in gathered_rewards.items()}

        if accelerator.is_local_main_process:
            print(f"Reward stats: max ({gathered_rewards['sds_per_step'].max()}) "
                  f"min ({gathered_rewards['sds_per_step'].min()}) "
                  f"mean({gathered_rewards['sds_per_step'].mean()})")

        if accelerator.is_main_process:
            wandb.log({
                "epoch": epoch,
                "reward_mean": gathered_rewards['ori_avg'].mean(),
                "reward_std": gathered_rewards['ori_avg'].std(),
                **{f"reward_{key}": value.mean() for key, value in gathered_rewards.items()},
            }, step=global_step)

        # Per-prompt stat tracking
        if config.per_prompt_stat_tracking:
            prompt_ids_all = accelerator.gather(samples["prompt_ids"]).cpu().numpy()
            prompts_all = tokenizers[0].batch_decode(prompt_ids_all, skip_special_tokens=True)
            advantages = stat_tracker.update(prompts_all, gathered_rewards['avg'])
            if accelerator.is_main_process:
                group_size, trained_prompt_num = stat_tracker.get_stats()
                zero_std_ratio, reward_std_mean = calculate_zero_std_ratio(prompts_all, gathered_rewards)
                wandb.log({
                    "group_size": group_size, "trained_prompt_num": trained_prompt_num,
                    "zero_std_ratio": zero_std_ratio, "reward_std_mean": reward_std_mean,
                }, step=global_step)
            stat_tracker.clear()
        else:
            advantages = (gathered_rewards['avg'] - gathered_rewards['avg'].mean()) / (gathered_rewards['avg'].std() + 1e-4)

        advantages = torch.as_tensor(advantages)
        samples["advantages"] = (
            advantages.reshape(accelerator.num_processes, -1, advantages.shape[-1])[accelerator.process_index]
            .to(accelerator.device)
        )
        if accelerator.is_local_main_process:
            print("advantages: ", samples["advantages"].abs().mean())

        del samples["rewards"]
        del samples["prompt_ids"]

        # Mask zero-advantage examples
        mask = (samples["advantages"].abs().sum(dim=1) != 0)
        num_batches = config.sample.num_batches_per_epoch
        true_count = mask.sum()
        if true_count % num_batches != 0:
            false_indices = torch.where(~mask)[0]
            num_to_change = num_batches - (true_count % num_batches)
            if len(false_indices) >= num_to_change:
                random_indices = torch.randperm(len(false_indices))[:num_to_change]
                mask[false_indices[random_indices]] = True
        if accelerator.is_main_process:
            wandb.log({"actual_batch_size": mask.sum().item() // num_batches}, step=global_step)
        samples = {k: v[mask] for k, v in samples.items()}

        total_batch_size, num_timesteps = samples["timesteps"].shape
        assert num_timesteps == config.sample.num_steps

        # ========== TRAINING ==========
        for inner_epoch in range(config.train.num_inner_epochs):
            perm = torch.randperm(total_batch_size, device=accelerator.device)
            samples = {k: v[perm] for k, v in samples.items()}

            samples_batched = {
                k: v.reshape(-1, total_batch_size // config.sample.num_batches_per_epoch, *v.shape[1:])
                for k, v in samples.items()
            }
            samples_batched = [dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())]

            pipeline.unet.train()
            info = defaultdict(list)
            for i, sample in tqdm(
                list(enumerate(samples_batched)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=not accelerator.is_local_main_process,
            ):
                train_timesteps = [step_index for step_index in range(int(num_train_timesteps * 0.5), num_train_timesteps)]
                for j in tqdm(
                    train_timesteps,
                    desc="Timestep",
                    position=1,
                    leave=False,
                    disable=not accelerator.is_local_main_process,
                ):
                    with accelerator.accumulate(unet):
                        with autocast():
                            prev_sample, log_prob, prev_sample_mean, std_dev_t = compute_log_prob_sdxl(
                                unet, ddpm_scheduler, sample, j,
                                sample["prompt_embeds"], sample["pooled_prompt_embeds"],
                                add_time_ids,
                                train_neg_prompt_embeds[:len(sample["prompt_embeds"])] if config.train.cfg else None,
                                train_neg_pooled_prompt_embeds[:len(sample["pooled_prompt_embeds"])] if config.train.cfg else None,
                                neg_add_time_ids if config.train.cfg else None,
                                config,
                            )
                            if config.train.beta > 0:
                                with torch.no_grad():
                                    with unet.module.disable_adapter():
                                        _, _, prev_sample_mean_ref, _ = compute_log_prob_sdxl(
                                            unet, ddpm_scheduler, sample, j,
                                            sample["prompt_embeds"], sample["pooled_prompt_embeds"],
                                            add_time_ids,
                                            train_neg_prompt_embeds[:len(sample["prompt_embeds"])] if config.train.cfg else None,
                                            train_neg_pooled_prompt_embeds[:len(sample["pooled_prompt_embeds"])] if config.train.cfg else None,
                                            neg_add_time_ids if config.train.cfg else None,
                                            config,
                                        )

                        # GRPO loss
                        advantages_j = torch.clamp(
                            sample["advantages"][:, j],
                            -config.train.adv_clip_max,
                            config.train.adv_clip_max,
                        )
                        ratio = torch.exp(log_prob - sample["log_probs"][:, j])
                        unclipped_loss = -advantages_j * ratio
                        clipped_loss = -advantages_j * torch.clamp(
                            ratio, 1.0 - config.train.clip_range, 1.0 + config.train.clip_range,
                        )
                        policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))

                        if config.train.beta > 0:
                            kl_loss = ((prev_sample_mean - prev_sample_mean_ref) ** 2).mean(dim=(1, 2, 3), keepdim=True) / (2 * std_dev_t ** 2)
                            kl_loss = torch.mean(kl_loss)
                            loss = policy_loss + config.train.beta * kl_loss
                        else:
                            loss = policy_loss

                        info["approx_kl"].append(0.5 * torch.mean((log_prob - sample["log_probs"][:, j]) ** 2))
                        info["clipfrac"].append(torch.mean((torch.abs(ratio - 1.0) > config.train.clip_range).float()))
                        info["clipfrac_gt_one"].append(torch.mean((ratio - 1.0 > config.train.clip_range).float()))
                        info["clipfrac_lt_one"].append(torch.mean((1.0 - ratio > config.train.clip_range).float()))
                        info["policy_loss"].append(policy_loss)
                        if config.train.beta > 0:
                            info["kl_loss"].append(kl_loss)
                        info["loss"].append(loss)
                        info["logp_rel_mean"].append(log_prob.mean())

                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(unet.parameters(), config.train.max_grad_norm)
                        optimizer.step()
                        optimizer.zero_grad()

                    if accelerator.sync_gradients:
                        info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                        info = accelerator.reduce(info, reduction="mean")
                        info.update({"epoch": epoch, "inner_epoch": inner_epoch})
                        if accelerator.is_main_process:
                            wandb.log(info, step=global_step)
                        global_step += 1
                        info = defaultdict(list)
                if config.train.ema:
                    ema.step(unet_trainable_parameters, global_step)

        epoch += 1


# --------------------- Evaluation ---------------------

def _eval_sdxl(
    pipeline, ddpm_scheduler, test_dataloader, tokenizers, text_encoders,
    config, accelerator, global_step, eval_reward_fn, executor,
    autocast, add_time_ids, neg_add_time_ids,
    neg_prompt_embeds, neg_pooled_prompt_embeds,
    num_reward_timesteps, ema, unet_trainable_parameters,
):
    if config.train.ema:
        ema.copy_ema_to(unet_trainable_parameters, store_temp=True)

    resolution = config.resolution
    all_rewards = defaultdict(list)
    last_batch_images_gather = None
    last_batch_prompts_gather = None

    for test_batch in tqdm(
        test_dataloader,
        desc="Eval: ",
        disable=not accelerator.is_local_main_process,
        position=0,
    ):
        prompts, prompt_metadata = test_batch
        prompt_embeds, pooled_prompt_embeds = encode_prompt_sdxl(
            tokenizers, text_encoders, prompts, accelerator.device,
        )

        eval_neg_pe = neg_prompt_embeds.repeat(len(prompts), 1, 1)
        eval_neg_ppe = neg_pooled_prompt_embeds.repeat(len(prompts), 1)

        eval_steps = getattr(config.sample, "eval_num_steps", config.sample.num_steps)
        with autocast():
            with torch.no_grad():
                all_latents, _, _ = sdxl_sample_with_logprob(
                    pipeline.unet, ddpm_scheduler,
                    prompt_embeds, pooled_prompt_embeds, add_time_ids,
                    neg_prompt_embeds=eval_neg_pe,
                    neg_pooled_prompt_embeds=eval_neg_ppe,
                    neg_add_time_ids=neg_add_time_ids,
                    num_inference_steps=eval_steps,
                    guidance_scale=config.sample.guidance_scale,
                    height=resolution, width=resolution,
                    device=accelerator.device,
                )

        x0 = all_latents[-1]
        with torch.no_grad():
            decoded = pipeline.vae.decode(
                x0.to(pipeline.vae.dtype) / pipeline.vae.config.scaling_factor
            ).sample
            images = (decoded / 2 + 0.5).clamp(0, 1)

        # External reward
        rewards = executor.submit(
            eval_reward_fn, images, prompts, prompt_metadata, only_strict=False,
        )
        time.sleep(0)
        rewards, _ = rewards.result()
        for key, value in rewards.items():
            rewards_gather = accelerator.gather(torch.as_tensor(value, device=accelerator.device)).cpu().numpy()
            all_rewards[key].append(rewards_gather)

        last_batch_images_gather = accelerator.gather(images).cpu().numpy()
        last_batch_prompt_ids = tokenizers[0](
            prompts, padding="max_length", max_length=256, truncation=True, return_tensors="pt",
        ).input_ids.to(accelerator.device)
        last_batch_prompt_ids_gather = accelerator.gather(last_batch_prompt_ids).cpu().numpy()
        last_batch_prompts_gather = tokenizers[0].batch_decode(last_batch_prompt_ids_gather, skip_special_tokens=True)

    all_rewards = {key: np.concatenate(value) for key, value in all_rewards.items()}
    if accelerator.is_main_process and last_batch_images_gather is not None:
        with tempfile.TemporaryDirectory() as tmpdir:
            num_samples = min(15, len(last_batch_images_gather))
            sample_indices = range(num_samples)
            for idx, index in enumerate(sample_indices):
                image = last_batch_images_gather[index]
                pil = Image.fromarray((image.transpose(1, 2, 0) * 255).astype(np.uint8))
                pil = pil.resize((resolution, resolution))
                pil.save(os.path.join(tmpdir, f"{idx}.jpg"))
            sampled_prompts = [last_batch_prompts_gather[index] for index in sample_indices]
            wandb.log({
                "eval_images": [
                    wandb.Image(os.path.join(tmpdir, f"{idx}.jpg"), caption=f"{prompt:.1000}")
                    for idx, prompt in enumerate(sampled_prompts)
                ],
                **{f"eval_reward_{key}": np.mean(value[value != -10]) for key, value in all_rewards.items()},
            }, step=global_step)

    if config.train.ema:
        ema.copy_temp_to(unet_trainable_parameters)


if __name__ == "__main__":
    app.run(main)
