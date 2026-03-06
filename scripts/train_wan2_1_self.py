from collections import defaultdict
import contextlib
import os
import datetime
from concurrent import futures
import time
import json
from absl import app, flags
from accelerate import Accelerator
from ml_collections import config_flags
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from diffusers import StableDiffusion3Pipeline, FlowMatchEulerDiscreteScheduler, WanPipeline
from diffusers.loaders import AttnProcsLayers
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
import numpy as np
import solace.prompts
import solace.rewards
from solace.stat_tracking import PerPromptStatTracker
from solace.diffusers_patch.wan_pipeline_with_logprob import wan_pipeline_with_logprob, sde_step_with_logprob
from solace.diffusers_patch.wan_prompt_embedding import encode_prompt
import torch
import wandb
from functools import partial
import tqdm
import tempfile
import itertools
from PIL import Image
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict, PeftModel
from peft.utils import get_peft_model_state_dict
import random
from torch.utils.data import Dataset, DataLoader, Sampler
from solace.ema import EMAModuleWrapper
import imageio

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")
logger = get_logger(__name__)


# --------------------- Datasets & Sampler ---------------------


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
        assert (
            self.total_samples % self.k == 0
        ), f"k can not div n*b, k{k}-num_replicas{num_replicas}-batch_size{batch_size}"
        self.m = self.total_samples // self.k
        self.epoch = 0

    def __iter__(self):
        while True:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)

            indices = torch.randperm(len(self.dataset), generator=g)[: self.m].tolist()
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


# --------------------- Helpers ---------------------


def compute_text_embeddings(prompt, text_encoders, tokenizers, max_sequence_length, device):
    with torch.no_grad():
        prompt_embeds = encode_prompt(text_encoders, tokenizers, prompt, max_sequence_length)
        prompt_embeds = prompt_embeds.to(device)
    return prompt_embeds


def set_adapter_and_freeze_params(transformer, adapter_name):
    transformer.module.set_adapter(adapter_name)
    for name, param in transformer.named_parameters():
        if "learner" in name:
            param.requires_grad_(True)
        elif "ref" in name:
            param.requires_grad_(False)


def calculate_zero_std_ratio(prompts, gathered_rewards):
    """
    Return:
      zero_std_ratio, mean_std_across_prompts
    Uses gathered_rewards['ori_avg'] (scalar per sample).
    """
    prompt_array = np.array(prompts)
    unique_prompts, inverse_indices, counts = np.unique(
        prompt_array, return_inverse=True, return_counts=True
    )

    grouped_rewards = gathered_rewards["ori_avg"][np.argsort(inverse_indices)]
    split_indices = np.cumsum(counts)[:-1]
    reward_groups = np.split(grouped_rewards, split_indices)

    prompt_std_devs = np.array([np.std(group) for group in reward_groups])
    zero_std_count = np.count_nonzero(prompt_std_devs == 0)
    zero_std_ratio = zero_std_count / len(prompt_std_devs)

    return zero_std_ratio, prompt_std_devs.mean()


def get_sigmas(noise_scheduler, timesteps, accelerator, n_dim=4, dtype=torch.float32):
    sigmas = noise_scheduler.sigmas.to(device=accelerator.device, dtype=dtype)
    schedule_timesteps = noise_scheduler.timesteps.to(accelerator.device)
    timesteps = timesteps.to(accelerator.device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


# --------------------- Logprob step (Wan) ---------------------


def compute_log_prob(transformer, pipeline, sample, j, embeds, negative_embeds, config, **kwargs):
    attention_kwargs = kwargs.get("attention_kwargs", getattr(config, "attention_kwargs", None))
    if config.train.cfg:
        noise_pred_text = transformer(
            hidden_states=sample["latents"][:, j],
            timestep=sample["timesteps"][:, j],
            encoder_hidden_states=embeds,
            attention_kwargs=attention_kwargs,
            return_dict=False,
        )[0]
        noise_pred_uncond = transformer(
            hidden_states=sample["latents"][:, j],
            timestep=sample["timesteps"][:, j],
            encoder_hidden_states=negative_embeds,
            attention_kwargs=attention_kwargs,
            return_dict=False,
        )[0]
        noise_pred = noise_pred_uncond + config.sample.guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )
    else:
        noise_pred = transformer(
            hidden_states=sample["latents"][:, j],
            timestep=sample["timesteps"][:, j],
            encoder_hidden_states=embeds,
            return_dict=False,
        )[0]

    prev_sample, log_prob, prev_sample_mean, std_dev_t, dt = sde_step_with_logprob(
        pipeline.scheduler,
        noise_pred.float(),
        sample["timesteps"][:, j],
        sample["latents"][:, j].float(),
        prev_sample=sample["next_latents"][:, j].float(),
        return_dt_and_std_dev_t=True,
    )
    return prev_sample, log_prob, prev_sample_mean, std_dev_t, dt


# --------------------- SDS self-certainty (Wan) ---------------------

def sds_self_confidence_scalar_wan(
    transformer,
    x0,                # [B, ...] arbitrary latent shape, e.g. [B,N,C] or [B,C,H,W]
    timesteps,         # [B, T_all] scheduler timesteps (ints)
    prompt_embeds,     # [B, S, D]
    neg_prompt_embed,  # [1, S, D]
    config,
    device,
    autocast_ctx,
    scheduler,
    use_steps=None,
):
    """
    Compute self-certainty scalar per sample using a flow-matching style probe.

    This is shape-agnostic: x0 is [B, ...] and we just treat all non-batch
    dimensions as one big vector when computing MSE.
    """
    B = x0.shape[0]
    latent_shape = x0.shape[1:]       # e.g. (N, C) or (C, H, W)
    sds_cfg = getattr(config.train, "sds", {})
    K = sds_cfg.get("k", 8)
    step_stride = sds_cfg.get("use_step_stride", 2)
    scale = sds_cfg.get("scale", 1.0)

    # how many timesteps to probe?
    T_all = timesteps.shape[1]
    T_used = min(use_steps if use_steps is not None else T_all, T_all)
    T_used = min(T_used, T_all - 1)  # skip final clean latent
    if T_used <= 0:
        return torch.zeros(B, device=device), None, None

    sds_per_step = torch.zeros(B, T_used, device=device, dtype=torch.float32)

    # Precompute sigma_t for each column j from the scheduler
    # We assume each row of `timesteps` is identical (as in your code).
    sched_ts = scheduler.timesteps.to(device)
    sched_sigmas = scheduler.sigmas.to(device=device, dtype=x0.dtype)

    t_row0 = timesteps[0].to(device)  # [T_all]
    sigma_per_col = []
    for t in t_row0[:T_used]:
        idx = (sched_ts == t).nonzero(as_tuple=False)
        if idx.numel() == 0:
            raise ValueError(f"timestep {t.item()} not found in scheduler.timesteps")
        sigma_per_col.append(sched_sigmas[idx[0, 0]])
    sigma_per_col = torch.stack(sigma_per_col, dim=0)  # [T_used]

    # We normally disable CFG for the probe
    use_cfg = False
    gs = float(config.sample.guidance_scale) if use_cfg else 1.0
    attention_kwargs = getattr(config, "attention_kwargs", None)

    # prompt embeddings: [B,S,D] -> [K*B,S,D]
    cond_pe = prompt_embeds.repeat(K, 1, 1)      # [K*B, S, D]
    neg_pe  = neg_prompt_embed.repeat(K * B, 1, 1)

    with torch.no_grad():
        with autocast_ctx():
            start_j = int(T_used * 0.5)          # focus on latter timesteps
            for j in range(start_j, T_used, step_stride):
                t_idx = timesteps[:, j] 
                # Current timestep index j → corresponding sigma_t
                sigma_t = sigma_per_col[j]            # scalar tensor
                # K symmetric probes: [K,B,...]
                eps = torch.randn(
                    (K // 2, B, *latent_shape),
                    device=device,
                    dtype=x0.dtype,
                )
                eps = torch.cat([eps, -eps], dim=0)   # [K,B,...]

                # x_t = (1 - sigma_t) * x0 + sigma_t * eps
                x0_for_eps = x0.unsqueeze(0)          # [1,B,...]
                xt = (1.0 - sigma_t) * x0_for_eps + sigma_t * eps   # [K,B,...]

                # flatten K and B for transformer: [K*B, ...]
                xt_flat = xt.reshape(K * B, *latent_shape)
                t_flat  = t_idx.repeat(K)             # [K*B]

                if use_cfg:
                    v_c = transformer(
                        hidden_states=xt_flat,
                        timestep=t_flat,
                        encoder_hidden_states=cond_pe,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0]
                    v_u = transformer(
                        hidden_states=xt_flat,
                        timestep=t_flat,
                        encoder_hidden_states=neg_pe,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0]
                    v_pred_flat = v_u + gs * (v_c - v_u)
                else:
                    v_pred_flat = transformer(
                        hidden_states=xt_flat,
                        timestep=t_flat,
                        encoder_hidden_states=cond_pe,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0]

                # replicate x0 along K: [K*B, ...]
                x0_flat = (
                    x0.unsqueeze(0)                   # [1,B,...]
                    .expand(K, *x0.shape)            # [K,B,...]
                    .reshape(K * B, *latent_shape)   # [K*B,...]
                )

                eps_flat = eps.reshape(K * B, *latent_shape)

                # FM-style: eps_hat = v + x0
                eps_hat_flat = v_pred_flat + x0_flat  # [K*B,...]

                # MSE over all non-batch dims
                reduce_dims = tuple(range(1, eps_hat_flat.ndim))
                mse_flat = torch.mean(
                    (eps_hat_flat - eps_flat) ** 2,
                    dim=reduce_dims,
                )                                     # [K*B]
                mse = mse_flat.view(K, B).mean(dim=0) # [B]

                sds_step = -torch.log(mse + 1e-6)     # higher = better
                sds_per_step[:, j] = sds_step

    # normalize per timestep across batch
    mean_t = sds_per_step.mean(dim=0, keepdim=True)          # [1,T]
    std_t  = sds_per_step.std(dim=0, keepdim=True).clamp_min(1e-6)
    sds_norm = (sds_per_step - mean_t) / std_t               # [B,T_used]

    # (optional) time weighting could go here, left disabled for now
    # float_t = timesteps[:, :T_used].float() / timesteps.max().float().clamp_min(1e-6)
    # sds_norm = sds_norm * (float_t * (1.0 - float_t))

    sds_scalar = scale * sds_norm.mean(dim=1)                # [B]
    return sds_scalar, sds_norm, sds_per_step


# --------------------- External-reward eval (unchanged) ---------------------


def eval(
    pipeline,
    test_dataloader,
    text_encoders,
    tokenizers,
    config,
    accelerator,
    global_step,
    reward_fn,
    executor,
    autocast,
    num_train_timesteps,
    ema,
    transformer_trainable_parameters,
):

    eval_seed = 42
    torch.manual_seed(eval_seed)
    np.random.seed(eval_seed)
    random.seed(eval_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(eval_seed)

    if accelerator.is_main_process:
        logger.info(f"Evaluation using deterministic seed: {eval_seed}")

    if config.train.ema:
        ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)
    neg_prompt_embed = compute_text_embeddings(
        [""], text_encoders, tokenizers, max_sequence_length=512, device=accelerator.device
    )

    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.test_batch_size, 1, 1)

    all_rewards = defaultdict(list)
    all_videos = []
    all_prompts = []
    for test_batch in tqdm(
        test_dataloader,
        desc="Eval: ",
        disable=not accelerator.is_local_main_process,
        position=0,
    ):
        prompts, prompt_metadata = test_batch
        prompt_embeds = compute_text_embeddings(
            prompts,
            text_encoders,
            tokenizers,
            max_sequence_length=512,
            device=accelerator.device,
        )
        if len(prompt_embeds) < len(sample_neg_prompt_embeds):
            sample_neg_prompt_embeds = sample_neg_prompt_embeds[: len(prompt_embeds)]

        with autocast():
            with torch.no_grad():
                videos, latents, log_probs, _ = wan_pipeline_with_logprob(
                    pipeline,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=sample_neg_prompt_embeds,
                    num_inference_steps=config.sample.eval_num_steps,
                    guidance_scale=config.sample.guidance_scale,
                    output_type="pt",
                    return_dict=False,
                    num_frames=config.frames,
                    height=config.height,
                    width=config.width,
                    determistic=True,
                )
        # rewards = executor.submit(
        #     reward_fn, videos, prompts, prompt_metadata, only_strict=False
        # )
        # time.sleep(0)
        # rewards, reward_metadata = rewards.result()

        # for key, value in rewards.items():
        #     rewards_gather = (
        #         accelerator.gather(torch.as_tensor(value, device=accelerator.device))
        #         .cpu()
        #         .numpy()
        #     )
        #     all_rewards[key].append(rewards_gather)

        # Collect ALL videos and prompts for logging
        videos_gather = accelerator.gather(torch.as_tensor(videos, device=accelerator.device)).cpu().numpy()
        prompt_ids = tokenizers[0](
            prompts, padding="max_length", max_length=512, truncation=True, return_tensors="pt"
        ).input_ids.to(accelerator.device)
        prompt_ids_gather = accelerator.gather(prompt_ids).cpu().numpy()
        prompts_gather = pipeline.tokenizer.batch_decode(prompt_ids_gather, skip_special_tokens=True)
        
        all_videos.append(videos_gather)
        all_prompts.extend(prompts_gather)

    # Concatenate all videos and rewards
    all_videos = np.concatenate(all_videos, axis=0)    
    # all_rewards = {key: np.concatenate(value) for key, value in all_rewards.items()}
    if accelerator.is_main_process:
        with tempfile.TemporaryDirectory() as tmpdir:
            num_samples = len(all_videos)
            print(f"Logging {num_samples} evaluation videos to wandb")

            for idx in range(num_samples):
                video = all_videos[idx].transpose(0, 2, 3, 1)
                frames = [img for img in video]
                frames = [(frame * 255).astype(np.uint8) for frame in frames]
                imageio.mimsave(
                    os.path.join(tmpdir, f"{idx}.mp4"),
                    frames,
                    fps=8,
                    codec="libx264",
                    format="FFMPEG",
                )

            # for key, value in all_rewards.items():
            #     print(key, value.shape)
            accelerator.log(
                {
                    "eval_images": [
                        wandb.Video(
                            os.path.join(tmpdir, f"{idx}.mp4"),
                            caption=f"{all_prompts[idx]:.1000}",
                            format="mp4",
                            fps=8,
                        )
                        for idx in range(num_samples)
                    ],
                },
                step=global_step,
            )
    if config.train.ema:
        ema.copy_temp_to(transformer_trainable_parameters)


# --------------------- Checkpoint helpers ---------------------


def unwrap_model(model, accelerator):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model


def save_ckpt(
    save_dir, transformer, global_step, accelerator, ema, transformer_trainable_parameters, config
):
    save_root = os.path.join(save_dir, "checkpoints", f"checkpoint-{global_step}")
    save_root_lora = os.path.join(save_root, "lora")
    os.makedirs(save_root_lora, exist_ok=True)
    if accelerator.is_main_process:
        if config.train.ema:
            ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)
        unwrap_model(transformer, accelerator).save_pretrained(save_root_lora)
        if config.train.ema:
            ema.copy_temp_to(transformer_trainable_parameters)


# --------------------- Main ---------------------


def main(_):
    config = FLAGS.config

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id

    if config.resume_from:
        config.resume_from = os.path.normpath(os.path.expanduser(config.resume_from))
        if "checkpoint_" not in os.path.basename(config.resume_from):
            checkpoints = list(
                filter(lambda x: "checkpoint_" in x, os.listdir(config.resume_from))
            )
            if len(checkpoints) == 0:
                raise ValueError(f"No checkpoints found in {config.resume_from}")
            config.resume_from = os.path.join(
                config.resume_from,
                sorted(checkpoints, key=lambda x: int(x.split("_")[-1]))[-1],
            )

    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)
    train_timesteps = [step_index for step_index in range(num_train_timesteps)]
    gradient_accumulation_steps = (
        config.train.gradient_accumulation_steps * (num_train_timesteps - num_train_timesteps // 2)
    )

    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )
    accelerator = Accelerator(
        log_with="wandb",
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )

    wandb_project_name = "wan_solace_sds"
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=wandb_project_name,
            config=config.to_dict(),
            init_kwargs={"wandb": {"name": config.run_name}},
        )
    logger.info(f"\n{config}")

    set_seed(config.seed, device_specific=True)

    # Load Wan pipeline
    pipeline = WanPipeline.from_pretrained(config.pretrained.model)
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.transformer.requires_grad_(not config.use_lora)

    text_encoders = [pipeline.text_encoder]
    tokenizers = [pipeline.tokenizer]

    pipeline.safety_checker = None
    pipeline.set_progress_bar_config(
        position=1,
        disable=not accelerator.is_local_main_process,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )

    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    pipeline.vae.to(accelerator.device, dtype=torch.float32)
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)

    if config.use_lora:
        pipeline.transformer.to(accelerator.device)
    else:
        pipeline.transformer.to(accelerator.device, dtype=inference_dtype)

    if config.use_lora:
        target_modules = [
            "add_k_proj",
            "add_q_proj",
            "add_v_proj",
            "to_add_out",
            "to_k",
            "to_out.0",
            "to_q",
            "to_v",
        ]
        transformer_lora_config = LoraConfig(
            r=32,
            lora_alpha=64,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        if config.train.lora_path:
            pipeline.transformer = PeftModel.from_pretrained(
                pipeline.transformer, config.train.lora_path
            )
            pipeline.transformer.set_adapter("default")
        else:
            pipeline.transformer = get_peft_model(
                pipeline.transformer, transformer_lora_config
            )

    transformer = pipeline.transformer
    transformer.enable_gradient_checkpointing()
    transformer_trainable_parameters = list(
        filter(lambda p: p.requires_grad, transformer.parameters())
    )
    ema = EMAModuleWrapper(
        transformer_trainable_parameters,
        decay=0.9,
        update_step_interval=8,
        device=accelerator.device,
    )

    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if config.train.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. `pip install bitsandbytes`"
            )
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        transformer_trainable_parameters,
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    eval_reward_fn = None
    # # External reward ONLY for eval
    # eval_reward_fn = getattr(solace.rewards, "multi_score")(
    #     accelerator.device, config.reward_fn
    # )

    # Datasets / loaders
    if config.prompt_fn == "general_ocr":
        train_dataset = TextPromptDataset(config.dataset, "train")
        test_dataset = TextPromptDataset(config.dataset, "test")

        train_sampler = DistributedKRepeatSampler(
            dataset=train_dataset,
            batch_size=config.sample.train_batch_size,
            k=config.sample.num_image_per_prompt,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            seed=42,
        )

        train_dataloader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=1,
            collate_fn=TextPromptDataset.collate_fn,
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=config.sample.test_batch_size,
            collate_fn=TextPromptDataset.collate_fn,
            shuffle=False,
            num_workers=8,
        )
    elif config.prompt_fn == "geneval":
        train_dataset = GenevalPromptDataset(config.dataset, "train")
        test_dataset = GenevalPromptDataset(config.dataset, "test")

        train_sampler = DistributedKRepeatSampler(
            dataset=train_dataset,
            batch_size=config.sample.train_batch_size,
            k=config.sample.num_image_per_prompt,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            seed=42,
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=1,
            collate_fn=GenevalPromptDataset.collate_fn,
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=config.sample.test_batch_size,
            collate_fn=GenevalPromptDataset.collate_fn,
            shuffle=False,
            num_workers=8,
        )
    else:
        raise NotImplementedError("Only general_ocr and geneval are supported with dataset")

    # Negative text embeddings for CFG
    neg_prompt_embed = compute_text_embeddings(
        [""], text_encoders, tokenizers, max_sequence_length=512, device=accelerator.device
    )

    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.train_batch_size, 1, 1)
    train_neg_prompt_embeds = neg_prompt_embed.repeat(
        config.train.batch_size * config.sample.sample_time_per_prompt, 1, 1
    )

    if (
        config.sample.num_image_per_prompt * config.sample.sample_time_per_prompt
        == 1
    ):
        config.per_prompt_stat_tracking = False

    if config.per_prompt_stat_tracking:
        stat_tracker = PerPromptStatTracker(config.sample.global_std)
    else:
        stat_tracker = None

    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast

    transformer, optimizer, train_dataloader, test_dataloader = accelerator.prepare(
        transformer, optimizer, train_dataloader, test_dataloader
    )

    executor = futures.ThreadPoolExecutor(max_workers=8)

    samples_per_epoch = (
        config.sample.train_batch_size
        * accelerator.num_processes
        * config.sample.num_batches_per_epoch
        * config.sample.sample_time_per_prompt
    )
    total_train_batch_size = (
        config.train.batch_size * accelerator.num_processes * config.train.gradient_accumulation_steps
    )

    logger.info("***** Running training (WAN + SDS self-certainty) *****")
    logger.info(f"  Num Epochs = {config.num_epochs}")
    logger.info(f"  Sample batch size per device = {config.sample.train_batch_size}")
    logger.info(f"  Train batch size per device  = {config.train.batch_size}")
    logger.info(f"  Grad Accum steps (x timesteps) = {config.train.gradient_accumulation_steps} x {num_train_timesteps}")
    logger.info(f"  Total samples/epoch = {samples_per_epoch}")
    logger.info(
        f"  GRPO updates/inner epoch = {samples_per_epoch // max(1, total_train_batch_size)}"
    )
    logger.info(f"  Inner epochs = {config.train.num_inner_epochs}")

    if config.resume_from:
        logger.info(f"Resuming from {config.resume_from}")
        accelerator.load_state(config.resume_from)
        first_epoch = int(config.resume_from.split("_")[-1]) + 1
    else:
        first_epoch = 0

    global_step = 0
    train_iter = iter(train_dataloader)

    for epoch in range(first_epoch, config.num_epochs):
        # =================== SAMPLING + SDS REWARD ===================
        pipeline.transformer.eval()
        samples = []
        prompts_last = None
        videos_last = None

        for i in tqdm(
            range(config.sample.num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            train_sampler.set_epoch(epoch * config.sample.num_batches_per_epoch + i)
            prompts, prompt_metadata = next(train_iter)

            prompt_embeds = compute_text_embeddings(
                prompts, text_encoders, tokenizers, max_sequence_length=512, device=accelerator.device
            )
            prompt_ids = tokenizers[0](
                prompts,
                padding="max_length",
                max_length=512,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(accelerator.device)

            # Eval & checkpointing on first sampling batch of epoch
            if i == 0 and epoch % config.eval_freq == 0:
                eval(
                    pipeline,
                    test_dataloader,
                    text_encoders,
                    tokenizers,
                    config,
                    accelerator,
                    global_step,
                    eval_reward_fn,
                    executor,
                    autocast,
                    num_train_timesteps,
                    ema,
                    transformer_trainable_parameters,
                )
            if i == 0 and epoch % config.save_freq == 0 and epoch > 0 and accelerator.is_main_process:
                save_ckpt(
                    config.save_dir,
                    transformer,
                    global_step,
                    accelerator,
                    ema,
                    transformer_trainable_parameters,
                    config,
                )

            # If you still want the "warmup" behavior for stat_tracking, keep this:
            if epoch < 2 and config.per_prompt_stat_tracking:
                continue

            for j in tqdm(
                range(config.sample.sample_time_per_prompt),
                desc=f"Epoch {epoch}: sampling | multi sample per prompt",
                disable=not accelerator.is_local_main_process,
                position=1,
            ):
                with autocast():
                    with torch.no_grad():
                        videos, latents, log_probs, kls = wan_pipeline_with_logprob(
                            pipeline,
                            prompt_embeds=prompt_embeds,
                            negative_prompt_embeds=sample_neg_prompt_embeds,
                            num_inference_steps=config.sample.num_steps,
                            guidance_scale=config.sample.guidance_scale,
                            output_type="pt",
                            return_dict=False,
                            num_frames=config.frames,
                            height=config.height,
                            width=config.width,
                            kl_reward=config.sample.kl_reward,
                        )

                latents = torch.stack(latents, dim=1)  # [B, T+1, C, H, W]
                log_probs = torch.stack(log_probs, dim=1)  # [B, T]
                kls = torch.stack(kls, dim=1)
                kl = kls.detach()

                timesteps = pipeline.scheduler.timesteps.repeat(
                    config.sample.train_batch_size, 1
                ).to(accelerator.device)

                x0 = latents[:, -1]  # clean latent

                # ---- SDS self-certainty per video sample ----
                sds_scalar, _, sds_per_step = sds_self_confidence_scalar_wan(
                    transformer=pipeline.transformer,
                    x0=x0,
                    timesteps=timesteps,
                    prompt_embeds=prompt_embeds,
                    neg_prompt_embed=neg_prompt_embed,
                    config=config,
                    device=accelerator.device,
                    autocast_ctx=autocast,
                    scheduler=pipeline.scheduler,
                    use_steps=num_train_timesteps,
                )

                samples.append(
                    {
                        "prompt_ids": prompt_ids,
                        "prompt_embeds": prompt_embeds,
                        "negative_prompt_embeds": sample_neg_prompt_embeds,
                        "timesteps": timesteps,
                        "latents": latents[:, :-1],
                        "next_latents": latents[:, 1:],
                        "log_probs": log_probs,
                        "kl": kl,
                        "sds_per_step": sds_per_step,
                        "rewards": {"avg": sds_scalar},
                    }
                )

                prompts_last = prompts
                videos_last = videos

        # For warmup case
        if epoch < 2 and config.per_prompt_stat_tracking:
            continue

        # Collate samples dict
        samples = {
            k: torch.cat([s[k] for s in samples], dim=0)
            if not isinstance(samples[0][k], dict)
            else {
                sub_key: torch.cat([s[k][sub_key] for s in samples], dim=0)
                for sub_key in samples[0][k]
            }
            for k in samples[0].keys()
        }

        # Optional: log a few videos with SDS scores every 10 epochs
        if epoch % 10 == 0 and accelerator.is_main_process and videos_last is not None:
            with tempfile.TemporaryDirectory() as tmpdir:
                num_samples_vis = min(15, len(videos_last))
                sample_indices_vis = random.sample(range(len(videos_last)), num_samples_vis)

                for idx, i_v in enumerate(sample_indices_vis):
                    video = videos_last[i_v]
                    frames = [img for img in video.cpu().numpy().transpose(0, 2, 3, 1)]
                    frames = [(frame * 255).astype(np.uint8) for frame in frames]
                    imageio.mimsave(
                        os.path.join(tmpdir, f"{idx}.mp4"),
                        frames,
                        fps=8,
                        codec="libx264",
                        format="FFMPEG",
                    )

                sampled_prompts = [prompts_last[i_v] for i_v in sample_indices_vis]
                sampled_rewards = [samples["rewards"]["avg"][i_v].item() for i_v in sample_indices_vis]

                accelerator.log(
                    {
                        "video": [
                            wandb.Video(
                                os.path.join(tmpdir, f"{idx}.mp4"),
                                caption=f"{prompt:.100} | sds: {avg_reward:.2f}",
                                format="mp4",
                            )
                            for idx, (prompt, avg_reward) in enumerate(
                                zip(sampled_prompts, sampled_rewards)
                            )
                        ],
                    },
                    step=global_step,
                )

        # Keep original values and per-step SDS for logging/stats
        samples["rewards"]["ori_avg"] = samples["rewards"]["avg"].clone()
        samples["rewards"]["sds_per_step"] = samples["sds_per_step"].clone()

        # Expand scalar SDS reward across timesteps for GRPO
        samples["rewards"]["avg"] = samples["rewards"]["avg"].unsqueeze(1).repeat(
            1, num_train_timesteps
        )

        gathered_rewards = {
            key: accelerator.gather(value) for key, value in samples["rewards"].items()
        }
        gathered_rewards = {key: value.cpu().numpy() for key, value in gathered_rewards.items()}

        accelerator.log(
            {
                "epoch": epoch,
                **{f"reward_{key}": value.mean() for key, value in gathered_rewards.items()},
                "kl": samples["kl"].mean().cpu().numpy(),
                "kl_abs": samples["kl"].abs().mean().cpu().numpy(),
            },
            step=global_step,
        )

        # Per-prompt stat tracking on SDS rewards
        if config.per_prompt_stat_tracking and stat_tracker is not None:
            prompt_ids_all = accelerator.gather(samples["prompt_ids"]).cpu().numpy()
            prompts_all = pipeline.tokenizer.batch_decode(
                prompt_ids_all, skip_special_tokens=True
            )

            advantages = stat_tracker.update(prompts_all, gathered_rewards["avg"])
            if accelerator.is_main_process:
                group_size, trained_prompt_num = stat_tracker.get_stats()
                zero_std_ratio, reward_std_mean = calculate_zero_std_ratio(
                    prompts_all, gathered_rewards
                )
                accelerator.log(
                    {
                        "group_size": group_size,
                        "trained_prompt_num": trained_prompt_num,
                        "zero_std_ratio": zero_std_ratio,
                        "reward_std_mean": reward_std_mean,
                    },
                    step=global_step,
                )
            stat_tracker.clear()
        else:
            # simple z-normalization
            avg_all = gathered_rewards["avg"]
            advantages = (avg_all - avg_all.mean()) / (avg_all.std() + 1e-4)

        advantages = torch.as_tensor(advantages)
        samples["advantages"] = (
            advantages.reshape(
                accelerator.num_processes, -1, advantages.shape[-1]
            )[accelerator.process_index].to(accelerator.device)
        )
        if accelerator.is_local_main_process:
            print("advantages: ", samples["advantages"].abs().mean())
            print("kl: ", samples["kl"].mean())

        del samples["rewards"]
        del samples["prompt_ids"]

        # Mask zero-advantage samples while keeping divisibility
        mask = samples["advantages"].abs().sum(dim=1) != 0
        num_batches = (
            config.sample.num_batches_per_epoch * config.sample.sample_time_per_prompt
        )
        true_count = mask.sum()
        if true_count == 0:
            print("advantages: ", samples["advantages"].abs().mean())
            print("mask.sum() == 0. revise in this rank")
            samples["advantages"] = samples["advantages"] + 1e-6
            print("after revise advantages: ", samples["advantages"].abs().mean())
            mask = samples["advantages"].abs().sum(dim=1) != 0

        if true_count % num_batches != 0:
            false_indices = torch.where(~mask)[0]
            num_to_change = num_batches - (true_count % num_batches)
            if len(false_indices) >= num_to_change:
                random_indices = torch.randperm(len(false_indices))[:num_to_change]
                mask[false_indices[random_indices]] = True

        accelerator.log(
            {
                "actual_batch_size": mask.sum().item()
                // (config.sample.num_batches_per_epoch * config.sample.sample_time_per_prompt),
            },
            step=global_step,
        )

        samples = {k: v[mask] for k, v in samples.items()}

        total_batch_size, num_timesteps_current = samples["timesteps"].shape
        assert num_timesteps_current == config.sample.num_steps

        # =================== TRAINING ===================
        for inner_epoch in range(config.train.num_inner_epochs):
            perm = torch.randperm(total_batch_size, device=accelerator.device)
            samples = {k: v[perm] for k, v in samples.items()}

            perms = torch.stack(
                [
                    torch.arange(num_timesteps_current, device=accelerator.device)
                    for _ in range(total_batch_size)
                ]
            )
            for key in ["timesteps", "latents", "next_latents", "log_probs"]:
                samples[key] = samples[key][
                    torch.arange(total_batch_size, device=accelerator.device)[:, None],
                    perms,
                ]

            micro_batch = total_batch_size // (
                config.sample.num_batches_per_epoch * config.sample.sample_time_per_prompt
            )

            samples_batched = {
                k: v.reshape(-1, micro_batch, *v.shape[1:]) for k, v in samples.items()
            }
            samples_batched = [
                dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
            ]

            pipeline.transformer.train()
            info = defaultdict(list)
            for i_batch, sample in tqdm(
                list(enumerate(samples_batched)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=not accelerator.is_local_main_process,
            ):
                if config.train.cfg:
                    embeds = sample["prompt_embeds"]
                    negative_embeds = train_neg_prompt_embeds[: len(sample["prompt_embeds"])]
                else:
                    embeds = sample["prompt_embeds"]
                    negative_embeds = None

                for j_t in tqdm(
                    train_timesteps[len(train_timesteps) // 2 : ],
                    desc="Timestep",
                    position=1,
                    leave=False,
                    disable=not accelerator.is_local_main_process,
                ):
                    with accelerator.accumulate(transformer):
                        with autocast():
                            (
                                prev_sample,
                                log_prob,
                                prev_sample_mean,
                                std_dev_t,
                                dt,
                            ) = compute_log_prob(
                                transformer,
                                pipeline,
                                sample,
                                j_t,
                                embeds,
                                negative_embeds,
                                config,
                            )
                            if config.train.beta > 0:
                                with torch.no_grad():
                                    with transformer.module.disable_adapter():
                                        (
                                            prev_sample_ref,
                                            log_prob_ref,
                                            prev_sample_mean_ref,
                                            std_dev_t_ref,
                                            dt_ref,
                                        ) = compute_log_prob(
                                            transformer,
                                            pipeline,
                                            sample,
                                            j_t,
                                            embeds,
                                            negative_embeds,
                                            config,
                                        )

                        advantages_j = torch.clamp(
                            sample["advantages"][:, j_t],
                            -config.train.adv_clip_max,
                            config.train.adv_clip_max,
                        )
                        ratio = torch.exp(log_prob - sample["log_probs"][:, j_t])
                        unclipped_loss = -advantages_j * ratio
                        clipped_loss = -advantages_j * torch.clamp(
                            ratio,
                            1.0 - config.train.clip_range,
                            1.0 + config.train.clip_range,
                        )
                        policy_loss = torch.mean(
                            torch.maximum(unclipped_loss, clipped_loss)
                        )

                        if config.train.beta > 0:
                            diff = prev_sample_mean - prev_sample_mean_ref   # [B, ...]
                            reduce_dims = tuple(range(1, diff.ndim))         # all non-batch dims
                            kl_per_sample = diff.pow(2).mean(dim=reduce_dims, keepdim=True) / (
                                2 * (std_dev_t * dt_ref) ** 2
                            )                                                # [B,1,...] effectively
                            kl_loss = kl_per_sample.mean()
                            loss = policy_loss + config.train.beta * kl_loss
                        else:
                            loss = policy_loss

                        info["approx_kl"].append(
                            0.5 * torch.mean((log_prob - sample["log_probs"][:, j_t]) ** 2)
                        )
                        info["clipfrac"].append(
                            torch.mean(
                                (torch.abs(ratio - 1.0) > config.train.clip_range).float()
                            )
                        )
                        info["policy_loss"].append(policy_loss)
                        if config.train.beta > 0:
                            info["kl_loss"].append(kl_loss)
                        info["loss"].append(loss)

                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(
                                transformer.parameters(), config.train.max_grad_norm
                            )
                        optimizer.step()
                        optimizer.zero_grad()

                    if accelerator.sync_gradients:
                        info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                        info = accelerator.reduce(info, reduction="mean")
                        info.update({"epoch": epoch, "inner_epoch": inner_epoch})
                        accelerator.log(info, step=global_step)
                        global_step += 1
                        info = defaultdict(list)

                if config.train.ema:
                    ema.step(transformer_trainable_parameters, global_step)


if __name__ == "__main__":
    app.run(main)
