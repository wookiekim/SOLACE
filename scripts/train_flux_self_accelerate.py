# Train Flux with INTERNAL (SDS-style) rewards only
# (Evaluation remains based on EXTERNAL rewards; eval() is kept as in your Flux script.)

from collections import defaultdict
import contextlib
import os
import datetime
from concurrent import futures
import time
import json
import hashlib
from absl import app, flags
from accelerate import Accelerator
from ml_collections import config_flags
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from diffusers import FluxPipeline
from diffusers.utils.torch_utils import is_compiled_module
import numpy as np
import solace.prompts
import solace.rewards  # kept for EVAL ONLY
from solace.stat_tracking import PerPromptStatTracker
from solace.diffusers_patch.flux_pipeline_with_logprob import pipeline_with_logprob
from solace.diffusers_patch.sd3_sde_with_logprob import sde_step_with_logprob
from solace.diffusers_patch.train_dreambooth_lora_flux import encode_prompt
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
        ), f"k can not divide n*b, k{k}-num_replicas{num_replicas}-batch_size{batch_size}"
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
        prompt_embeds, pooled_prompt_embeds, _ = encode_prompt(
            text_encoders, tokenizers, prompt, max_sequence_length
        )
        prompt_embeds = prompt_embeds.to(device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device)
    return prompt_embeds, pooled_prompt_embeds


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
        h = hashlib.sha256(prompt.encode()).digest()
        prompt_hash_int = int.from_bytes(h[:4], 'big')
        seed = (base_seed + prompt_hash_int) % (2**31)
        gen = torch.Generator().manual_seed(seed)
        generators.append(gen)
    return generators


def unwrap_model(model, accelerator):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model


def save_ckpt(save_dir, transformer, global_step, accelerator, ema, transformer_trainable_parameters, config):
    save_root = os.path.join(save_dir, "checkpoints", f"checkpoint-{global_step}")
    save_root_lora = os.path.join(save_root, "lora")
    os.makedirs(save_root_lora, exist_ok=True)
    if accelerator.is_main_process:
        if config.train.ema:
            ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)
        unwrap_model(transformer, accelerator).save_pretrained(save_root_lora)
        if config.train.ema:
            ema.copy_temp_to(transformer_trainable_parameters)


# --------------------- PPO/GRPO log-prob (Flux) ---------------------

def compute_log_prob(transformer, pipeline, sample, j, config):
    packed_noisy_model_input = sample["latents"][:, j]
    device = packed_noisy_model_input.device
    dtype = packed_noisy_model_input.dtype

    if transformer.module.config.guidance_embeds:
        guidance = torch.tensor([config.sample.guidance_scale], device=device)
        guidance = guidance.expand(packed_noisy_model_input.shape[0])
    else:
        guidance = None

    model_pred = transformer(
        hidden_states=packed_noisy_model_input,
        timestep=sample["timesteps"][:, j] / 1000,   # normalized
        guidance=guidance,
        pooled_projections=sample["pooled_prompt_embeds"],
        encoder_hidden_states=sample["prompt_embeds"],
        txt_ids=torch.zeros(sample["prompt_embeds"].shape[1], 3, device=device, dtype=dtype),
        img_ids=sample["image_ids"][0],
        return_dict=False,
    )[0]

    prev_sample, log_prob, prev_sample_mean, std_dev_t = sde_step_with_logprob(
        pipeline.scheduler,
        model_pred.float(),
        sample["timesteps"][:, j],
        sample["latents"][:, j].float(),
        prev_sample=sample["next_latents"][:, j].float(),
        noise_level=config.sample.noise_level,
    )
    return prev_sample, log_prob, prev_sample_mean, std_dev_t


# --------------------- INTERNAL REWARD: SDS-style self-confidence (Flux) ---------------------

def sds_self_confidence_scalar_flux(
    transformer,
    x0,                        # [B, N, C] (Flux latent space: tokens x channels)
    timesteps,                 # [B, T_all] (ints like 999..)
    prompt_embeds,             # [B, S, D]
    pooled_prompt_embeds,      # [B, Dp]
    img_ids,                   # as returned by pipeline_with_logprob; pass through
    config,
    device,
    autocast_ctx,
    use_steps=None,
):
    """
    Returns:
      sds_scalar [B] (higher is better), sds_norm [B, T_used], sds_raw [B, T_used]
    """
    import torch

    # Resolve model dtype (handles PEFT as well)
    try:
        model_dtype = next(transformer.parameters()).dtype
    except StopIteration:
        model_dtype = torch.get_default_dtype()

    # Shapes & hyperparams
    B, N, C = x0.shape
    K = getattr(getattr(config, "train", {}), "sds", {}).get("k", 4)
    step_stride = getattr(getattr(config, "train", {}), "sds", {}).get("use_step_stride", 1)
    scale = getattr(getattr(config, "train", {}), "sds", {}).get("scale", 1.0)

    T_all = timesteps.shape[1]
    T_used = min(use_steps if use_steps is not None else T_all, T_all)
    T_used = min(T_used, T_all - 1)  # need at least one transition
    if T_used <= 0:
        return torch.zeros(B, device=device, dtype=torch.float32), None, None

    # Buffers (compute in fp32 for stability)
    sds_per_step = torch.zeros(B, T_used, device=device, dtype=torch.float32)

    # Cast inputs to model dtype
    x0_md   = x0.to(device=device, dtype=model_dtype)                  # [B, N, C]
    cond_pe = prompt_embeds.to(device=device, dtype=model_dtype)       # [B, S, D]
    cond_pp = pooled_prompt_embeds.to(device=device, dtype=model_dtype)# [B, Dp]

    # Repeat text conds to match K*B batch
    cond_pe = cond_pe.repeat(K, 1, 1)   # [K*B, S, D]
    cond_pp = cond_pp.repeat(K, 1)      # [K*B, Dp]

    with torch.no_grad():
        with autocast_ctx():
            # Probe only the latter half by default (tends to carry more signal)
            j_start = int(T_used * 0.6)
            for j in range(j_start, T_used, step_stride):
                # t normalization and broadcast
                t_idx   = timesteps[:, j].to(device=device)                      # [B]
                t_norm  = (t_idx.float() / 1000.0).to(dtype=model_dtype)         # [B]
                t_expand = t_norm.view(1, B, 1, 1)                                # [1, B, 1, 1]

                # K probes of Gaussian noise in model dtype
                eps = torch.randn(K, B, N, C, device=device, dtype=model_dtype)  # [K, B, N, C]

                # Simple FM mix: x_t = (1 - t) * x0 + t * eps
                x0_4d = x0_md.unsqueeze(0)                                       # [1, B, N, C]
                xt    = (1.0 - t_expand) * x0_4d + t_expand * eps                # [K, B, N, C]
                xt_flat = xt.reshape(K * B, N, C)                                 # [K*B, N, C]
                t_flat  = t_norm.repeat(K)                                        # [K*B]

                # Flux forward (no guidance)
                v_pred_flat = transformer(
                    hidden_states=xt_flat,
                    timestep=t_flat,
                    guidance=torch.full((xt_flat.shape[0],), 1.0, device=device, dtype=model_dtype),
                    encoder_hidden_states=cond_pe,
                    pooled_projections=cond_pp,
                    txt_ids=torch.zeros(prompt_embeds.shape[1], 3, device=device, dtype=model_dtype),
                    img_ids=img_ids,
                    return_dict=False,
                )[0]  # [K*B, N, C] in model_dtype

                # FM relation: eps_hat = v + x0
                x0_flat  = x0_md.unsqueeze(0).expand(K, -1, -1, -1).reshape(K * B, N, C)
                eps_flat = eps.reshape(K * B, N, C)

                # MSE in float32 for stability
                diff = (v_pred_flat + x0_flat - eps_flat).float()
                mse_flat = diff.pow(2).mean(dim=(1, 2))   # [K*B]
                mse = mse_flat.view(K, B).mean(dim=0)     # [B]
                sds_step = -torch.log(mse + 1e-6)         # higher is better
                sds_per_step[:, j] = sds_step

    # Per-timestep normalization across batch
    mean_t = sds_per_step.mean(dim=0, keepdim=True)
    std_t  = sds_per_step.std(dim=0, keepdim=True).clamp_min(1e-6)
    sds_norm = (sds_per_step - mean_t) / std_t  # [B, T_used]

    # Optional time weighting
    float_t = (timesteps[:, :T_used].to(device=device).float() / 1000.0)
    sds_norm = sds_norm * (float_t * (1.0 - float_t))

    # Aggregate to a single scalar per image
    sds_scalar = scale * sds_norm.mean(dim=1)  # [B]

    return sds_scalar, sds_norm, sds_per_step


# --------------------- EVAL (unchanged; uses EXTERNAL rewards) ---------------------

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
    if config.train.ema:
        ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)

    all_rewards = defaultdict(list)
    for test_batch in tqdm(
        test_dataloader,
        desc="Eval: ",
        disable=not accelerator.is_local_main_process,
        position=0,
    ):
        prompts, prompt_metadata = test_batch
        prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
            prompts, text_encoders, tokenizers, max_sequence_length=128, device=accelerator.device
        )
        with autocast():
            with torch.no_grad():
                images, _, _, _, _ = pipeline_with_logprob(
                    pipeline,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    num_inference_steps=config.sample.eval_num_steps,
                    guidance_scale=config.sample.guidance_scale,
                    output_type="pt",
                    height=config.resolution,
                    width=config.resolution,
                    noise_level=0,
                )
        rewards = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=False)
        time.sleep(0)
        rewards, reward_metadata = rewards.result()

        for key, value in rewards.items():
            rewards_gather = accelerator.gather(torch.as_tensor(value, device=accelerator.device)).cpu().numpy()
            all_rewards[key].append(rewards_gather)

    last_batch_images_gather = accelerator.gather(torch.as_tensor(images, device=accelerator.device)).cpu().numpy()
    last_batch_prompt_ids = tokenizers[0](
        prompts,
        padding="max_length",
        max_length=256,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(accelerator.device)
    last_batch_prompt_ids_gather = accelerator.gather(last_batch_prompt_ids).cpu().numpy()
    last_batch_prompts_gather = pipeline.tokenizer.batch_decode(
        last_batch_prompt_ids_gather, skip_special_tokens=True
    )
    last_batch_rewards_gather = {}
    for key, value in rewards.items():
        last_batch_rewards_gather[key] = accelerator.gather(torch.as_tensor(value, device=accelerator.device)).cpu().numpy()

    all_rewards = {key: np.concatenate(value) for key, value in all_rewards.items()}
    if accelerator.is_main_process:
        with tempfile.TemporaryDirectory() as tmpdir:
            num_samples = min(15, len(last_batch_images_gather))
            sample_indices = range(num_samples)
            for idx, index in enumerate(sample_indices):
                image = last_batch_images_gather[index]
                pil = Image.fromarray(
                    (image.transpose(1, 2, 0) * 255).astype(np.uint8)
                )
                pil = pil.resize((config.resolution, config.resolution))
                pil.save(os.path.join(tmpdir, f"{idx}.jpg"))
            sampled_prompts = [last_batch_prompts_gather[index] for index in sample_indices]
            sampled_rewards = [
                {k: last_batch_rewards_gather[k][index] for k in last_batch_rewards_gather}
                for index in sample_indices
            ]
            for key, value in all_rewards.items():
                print(key, value.shape)
            wandb.log(
                {
                    "eval_images": [
                        wandb.Image(
                            os.path.join(tmpdir, f"{idx}.jpg"),
                            caption=f"{prompt:.1000} | "
                            + " | ".join(f"{k}: {v:.2f}" for k, v in reward.items() if v != -10),
                        )
                        for idx, (prompt, reward) in enumerate(zip(sampled_prompts, sampled_rewards))
                    ],
                    **{
                        f"eval_reward_{key}": np.mean(value[value != -10])
                        for key, value in all_rewards.items()
                    },
                },
                step=global_step,
            )
    if config.train.ema:
        ema.copy_temp_to(transformer_trainable_parameters)


# --------------------- Main ---------------------

def main(_):
    # basic Accelerate and logging setup
    config = FLAGS.config

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

    # NOTE: pure Accelerate (no DeepSpeed plugin). If your `accelerate config` file
    # specifies DeepSpeed, you should regenerate it without DeepSpeed.
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps * max(1, int(0.6 * num_train_timesteps)),
    )
    if accelerator.is_main_process:
        wandb.init(project="solace")
    logger.info(f"\n{config}")

    # seed
    set_seed(config.seed, device_specific=True)

    # load pipeline
    pipeline = FluxPipeline.from_pretrained(
        config.pretrained.model,
        low_cpu_mem_usage=False,
    )
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    pipeline.transformer.requires_grad_(not config.use_lora)

    text_encoders = [pipeline.text_encoder, pipeline.text_encoder_2]
    tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2]

    pipeline.safety_checker = None
    pipeline.set_progress_bar_config(
        position=1,
        disable=not accelerator.is_local_main_process,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )

    # dtype
    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    pipeline.vae.to(accelerator.device, dtype=torch.float32)
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder_2.to(accelerator.device, dtype=inference_dtype)
    pipeline.transformer.to(accelerator.device)

    # LoRA
    if config.use_lora:
        target_modules = [
            "attn.to_k",
            "attn.to_q",
            "attn.to_v",
            "attn.to_out.0",
            "attn.add_k_proj",
            "attn.add_q_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "ff.net.0.proj",
            "ff.net.2",
            "ff_context.net.0.proj",
            "ff_context.net.2",
        ]
        transformer_lora_config = LoraConfig(
            r=64,
            lora_alpha=128,
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

    # optimizer
    if config.train.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. "
                "You can do so by running `pip install bitsandbytes`"
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

    # EVAL reward fns (external) & executor — now simple, no DeepSpeed hooks
    reward_fn = getattr(solace.rewards, "multi_score")(accelerator.device, config.reward_fn)
    eval_reward_fn = getattr(solace.rewards, "multi_score")(accelerator.device, config.reward_fn)
    executor = futures.ThreadPoolExecutor(max_workers=8)

    # datasets/loaders
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

    # per-prompt tracking toggle
    if config.sample.num_image_per_prompt == 1:
        config.per_prompt_stat_tracking = False
    if config.per_prompt_stat_tracking:
        stat_tracker = PerPromptStatTracker(config.sample.global_std)

    # autocast choice
    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast

    # prepare model, optimizer, dataloaders
    transformer, optimizer, train_dataloader, test_dataloader = accelerator.prepare(
        transformer, optimizer, train_dataloader, test_dataloader
    )

    # training bookkeeping
    samples_per_epoch = (
        config.sample.train_batch_size
        * accelerator.num_processes
        * config.sample.num_batches_per_epoch
    )
    total_train_batch_size = (
        config.train.batch_size
        * accelerator.num_processes
        * config.train.gradient_accumulation_steps
    )

    logger.info("***** Running training (INTERNAL rewards only: SDS) *****")
    logger.info(f"  Sample batch size per device = {config.sample.train_batch_size}")
    logger.info(f"  Train batch size per device  = {config.train.batch_size}")
    logger.info(
        f"  Grad Accum steps (x timesteps) = "
        f"{config.train.gradient_accumulation_steps} x {max(1, num_train_timesteps)}"
    )
    logger.info(f"  Total samples/epoch = {samples_per_epoch}")
    logger.info(
        f"  GRPO updates/inner epoch = "
        f"{samples_per_epoch // max(1, total_train_batch_size)}"
    )
    logger.info(f"  Inner epochs = {config.train.num_inner_epochs}")

    epoch = 0
    global_step = 0
    train_iter = iter(train_dataloader)

    while True:
        #################### EVAL (external rewards kept) ####################
        pipeline.transformer.eval()
        if epoch % config.eval_freq == 0:
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
        if epoch % config.save_freq == 0 and epoch > 0 and accelerator.is_main_process:
            save_ckpt(
                config.save_dir,
                transformer,
                global_step,
                accelerator,
                ema,
                transformer_trainable_parameters,
                config,
            )

        #################### SAMPLING + INTERNAL REWARD (SDS) ####################
        pipeline.transformer.eval()
        samples = []
        last_images = None
        last_prompts = None
        for i in tqdm(
            range(config.sample.num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            train_sampler.set_epoch(epoch * config.sample.num_batches_per_epoch + i)
            prompts, prompt_metadata = next(train_iter)

            prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
                prompts, text_encoders, tokenizers, max_sequence_length=128, device=accelerator.device
            )
            prompt_ids = tokenizers[0](
                prompts,
                padding="max_length",
                max_length=256,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(accelerator.device)

            generator = (
                create_generator(prompts, base_seed=epoch * 10000 + i)
                if config.sample.same_latent
                else None
            )
            with autocast():
                with torch.no_grad():
                    images, latents, image_ids, text_ids, log_probs = pipeline_with_logprob(
                        pipeline,
                        prompt_embeds=prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        num_inference_steps=config.sample.num_steps,
                        guidance_scale=config.sample.guidance_scale,
                        output_type="pt",
                        height=config.resolution,
                        width=config.resolution,
                        noise_level=config.sample.noise_level,
                        generator=generator,
                    )

            latents = torch.stack(latents, dim=1)   # (B, T+1, N, C) in Flux latent space
            log_probs = torch.stack(log_probs, dim=1)  # (B, T)
            timesteps = pipeline.scheduler.timesteps.repeat(
                config.sample.train_batch_size, 1
            ).to(accelerator.device)

            # clean latent (x0) from trajectory
            x0 = latents[:, -1]

            # INTERNAL SDS reward per image
            sds_scalar, sds_norm, sds_per_step = sds_self_confidence_scalar_flux(
                transformer=pipeline.transformer,  # non-DDP model for probing
                x0=x0,
                timesteps=timesteps,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                img_ids=image_ids,
                config=config,
                device=accelerator.device,
                autocast_ctx=autocast,
                use_steps=num_train_timesteps,
            )

            samples.append(
                {
                    "prompt_ids": prompt_ids,
                    "prompt_embeds": prompt_embeds,
                    "pooled_prompt_embeds": pooled_prompt_embeds,
                    "image_ids": image_ids.unsqueeze(0).repeat(len(prompt_ids), 1, 1),
                    "timesteps": timesteps,
                    "latents": latents[:, :-1],       # latent before t
                    "next_latents": latents[:, 1:],   # latent after t
                    "log_probs": log_probs,
                    # stuff GRPO expects
                    "rewards": {"avg": sds_scalar},
                    "sds_per_step": (
                        sds_per_step if sds_per_step is not None else torch.zeros_like(log_probs)
                    ),
                }
            )

            last_images = images
            last_prompts = prompts

        # Collate to big tensors
        samples = {
            k: torch.cat([s[k] for s in samples], dim=0)
            if not isinstance(samples[0][k], dict)
            else {
                sub_key: torch.cat([s[k][sub_key] for s in samples], dim=0)
                for sub_key in samples[0][k]
            }
            for k in samples[0].keys()
        }

        # (Optional) log a small grid
        if epoch % 10 == 0 and accelerator.is_main_process and last_images is not None:
            with tempfile.TemporaryDirectory() as tmpdir:
                num_samples = min(15, len(last_images))
                sample_indices = random.sample(range(len(last_images)), num_samples)
                for idx, ii in enumerate(sample_indices):
                    image = last_images[ii]
                    pil = Image.fromarray(
                        (image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                    )
                    pil = pil.resize((config.resolution, config.resolution))
                    pil.save(os.path.join(tmpdir, f"{idx}.jpg"))
                sampled_prompts = [last_prompts[ii] for ii in sample_indices]
                sampled_rewards = [samples["rewards"]["avg"][ii].item() for ii in sample_indices]
                wandb.log(
                    {
                        "images": [
                            wandb.Image(
                                os.path.join(tmpdir, f"{idx}.jpg"),
                                caption=f"{prompt:.100} | sds: {avg_reward:.2f}",
                            )
                            for idx, (prompt, avg_reward) in enumerate(
                                zip(sampled_prompts, sampled_rewards)
                            )
                        ],
                    },
                    step=global_step,
                )

        # Keep original, and expand for GRPO across time
        samples["rewards"]["ori_avg"] = samples["rewards"]["avg"].clone()
        samples["rewards"]["sds_per_step"] = samples["sds_per_step"].clone()
        samples["rewards"]["avg"] = samples["rewards"]["avg"].unsqueeze(1).repeat(
            1, num_train_timesteps
        )

        # Gather across processes (for stat/adv computation)
        gathered_rewards = {key: accelerator.gather(value) for key, value in samples["rewards"].items()}
        gathered_rewards = {key: value.cpu().numpy() for key, value in gathered_rewards.items()}

        if accelerator.is_local_main_process:
            print(
                f"SDS stats: max ({gathered_rewards['sds_per_step'].max()}) "
                f"min ({gathered_rewards['sds_per_step'].min()}) "
                f"mean({gathered_rewards['sds_per_step'].mean()})"
            )

        if accelerator.is_main_process:
            wandb.log(
                {"epoch": epoch, **{f"reward_{key}": value.mean() for key, value in gathered_rewards.items()}},
                step=global_step,
            )

        # Per-prompt stat tracking on SDS
        if config.per_prompt_stat_tracking:
            prompt_ids_all = accelerator.gather(samples["prompt_ids"]).cpu().numpy()
            prompts_all = pipeline.tokenizer.batch_decode(prompt_ids_all, skip_special_tokens=True)
            advantages = stat_tracker.update(prompts_all, gathered_rewards["avg"])
            if accelerator.is_main_process:
                group_size, trained_prompt_num = stat_tracker.get_stats()
                zero_std_ratio, reward_std_mean = calculate_zero_std_ratio(
                    prompts_all, gathered_rewards
                )
                wandb.log(
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
            advantages = (
                gathered_rewards["avg"] - gathered_rewards["avg"].mean()
            ) / (gathered_rewards["avg"].std() + 1e-4)

        # Ungather back to local shard
        advantages = torch.as_tensor(advantages)
        samples["advantages"] = (
            advantages.reshape(
                accelerator.num_processes, -1, advantages.shape[-1]
            )[accelerator.process_index].to(accelerator.device)
        )

        if accelerator.is_local_main_process:
            print("advantages: ", samples["advantages"].abs().mean())

        # cleanup keys not needed for training loop
        del samples["rewards"]
        del samples["prompt_ids"]

        total_batch_size, num_timesteps = samples["timesteps"].shape
        assert num_timesteps == config.sample.num_steps

        #################### TRAINING (GRPO with SDS-derived advantages) ####################
        for inner_epoch in range(config.train.num_inner_epochs):
            # shuffle along batch dimension
            perm = torch.randperm(total_batch_size, device=accelerator.device)
            samples = {k: v[perm] for k, v in samples.items()}

            # rebatch
            samples_batched = {
                k: v.reshape(
                    -1, total_batch_size // config.sample.num_batches_per_epoch, *v.shape[1:]
                )
                for k, v in samples.items()
            }
            samples_batched = [
                dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
            ]

            pipeline.transformer.train()
            info = defaultdict(list)
            for i, sample in tqdm(
                list(enumerate(samples_batched)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=not accelerator.is_local_main_process,
            ):
                # Train across timesteps (you may restrict to the latter half if desired)
                train_timesteps = [
                    step_index
                    for step_index in range(int(0.6 * num_train_timesteps), num_train_timesteps)
                ]
                for j in tqdm(
                    train_timesteps,
                    desc="Timestep",
                    position=1,
                    leave=False,
                    disable=not accelerator.is_local_main_process,
                ):
                    with accelerator.accumulate(transformer):
                        with autocast():
                            prev_sample, log_prob, prev_sample_mean, std_dev_t = compute_log_prob(
                                transformer, pipeline, sample, j, config
                            )
                            if config.train.beta > 0:
                                with torch.no_grad():
                                    with transformer.module.disable_adapter():
                                        _, _, prev_sample_mean_ref, _ = compute_log_prob(
                                            transformer, pipeline, sample, j, config
                                        )

                        advantages_j = torch.clamp(
                            sample["advantages"][:, j],
                            -config.train.adv_clip_max,
                            config.train.adv_clip_max,
                        )
                        ratio = torch.exp(log_prob - sample["log_probs"][:, j])
                        unclipped_loss = -advantages_j * ratio
                        clipped_loss = -advantages_j * torch.clamp(
                            ratio,
                            1.0 - config.train.clip_range,
                            1.0 + config.train.clip_range,
                        )
                        policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
                        if config.train.beta > 0:
                            # NOTE: Flux latent has 3 dims (N,C,H/W collapsed depending on your implementation);
                            # we use dims=(1,2,3) as in your original code.
                            kl_loss = (
                                (prev_sample_mean - prev_sample_mean_ref) ** 2
                            ).mean(dim=(1, 2), keepdim=True) / (2 * std_dev_t ** 2)
                            kl_loss = torch.mean(kl_loss)
                            loss = policy_loss + config.train.beta * kl_loss
                        else:
                            loss = policy_loss

                        info["approx_kl"].append(
                            0.5 * torch.mean((log_prob - sample["log_probs"][:, j]) ** 2)
                        )
                        info["clipfrac"].append(
                            torch.mean(
                                (torch.abs(ratio - 1.0) > config.train.clip_range).float()
                            )
                        )
                        info["clipfrac_gt_one"].append(
                            torch.mean((ratio - 1.0 > config.train.clip_range).float())
                        )
                        info["clipfrac_lt_one"].append(
                            torch.mean((1.0 - ratio > config.train.clip_range).float())
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
                        if accelerator.is_main_process:
                            wandb.log(info, step=global_step)
                        global_step += 1
                        info = defaultdict(list)
                if config.train.ema:
                    ema.step(transformer_trainable_parameters, global_step)

        epoch += 1


if __name__ == "__main__":
    app.run(main)
