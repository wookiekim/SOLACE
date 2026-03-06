import ml_collections
import imp
import os

base = imp.load_source("base", os.path.join(os.path.dirname(__file__), "base.py"))

def compressibility():
    config = base.get_config()

    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    config.use_lora = True

    config.sample.batch_size = 8
    config.sample.num_batches_per_epoch = 4

    config.train.batch_size = 4
    config.train.gradient_accumulation_steps = 2

    # prompting
    config.prompt_fn = "general_ocr"

    # rewards
    config.reward_fn = {"jpeg_compressibility": 1}
    config.per_prompt_stat_tracking = True
    return config


# ===================== WAN 2.1 Video Config =====================

def general_ocr_wan2_1():
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/video_dataset")

    config.pretrained.model = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    config.sample.num_steps = 14
    config.sample.eval_num_steps = 50
    config.sample.guidance_scale=5.0
    config.run_name = "wan_solace"

    config.height = 480
    config.width = 832
    config.frames = 81
    config.sample.train_batch_size = 2
    config.sample.num_image_per_prompt = 4
    config.sample.num_batches_per_epoch = 1
    config.sample.sample_time_per_prompt = 1
    config.sample.test_batch_size = 2

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = 4
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    # kl loss
    config.train.beta = 0.004
    config.train.learning_rate = 1e-4
    config.train.clip_range=1e-3
    # KL reward and KL loss are two ways to incorporate KL divergence. KL reward adds KL to the reward,
    # while KL loss, introduced by GRPO, directly adds KL loss to the policy loss.
    # We support both methods, but KL loss is recommended as the preferred option.
    config.sample.kl_reward = 0
    config.train.sft=0.0
    config.train.sft_batch_size=3
    # Whether to use the std of all samples or the current group's.
    config.sample.global_std=False
    config.train.ema=True
    config.mixed_precision = "bf16"
    config.diffusion_loss = True
    config.num_epochs = 100000
    config.save_freq = 25
    config.eval_freq = 25
    config.save_dir = 'logs/video_ocr/wan_solace'
    config.resume_from = None
    config.reward_fn = {
        "video_ocr": 1.0,
    }

    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config


# ===================== SD3.5 Configs =====================

def general_ocr_sd3_8gpu():
    gpu_number = 8
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/ocr")

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 7.0

    config.resolution = 512
    config.sample.train_batch_size = 8
    config.sample.num_image_per_prompt = 16
    config.sample.num_batches_per_epoch = 4
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number!"
    config.sample.test_batch_size = 16

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    # kl loss
    config.train.beta = 0.04
    # Whether to use the std of all samples or the current group's.
    config.sample.global_std = True
    config.sample.same_latent = False
    config.train.ema = True
    # A large num_epochs is intentionally set here. Training will be manually stopped once sufficient
    config.save_freq = 60
    config.eval_freq = 60
    config.save_dir = 'logs/ocr/sd3_self_8gpu'
    config.reward_fn = {
        "ocr": 1.0,
    }

    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config


def general_ocr_sd3_8gpu_1024():
    gpu_number = 8
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/ocr")

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 7.0

    config.resolution = 1024
    config.sample.train_batch_size = 2
    config.sample.num_image_per_prompt = 16
    config.sample.num_batches_per_epoch = 16
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number!"
    config.sample.test_batch_size = 16

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    # kl loss
    config.train.beta = 0.04
    # Whether to use the std of all samples or the current group's.
    config.sample.global_std = True
    config.sample.same_latent = False
    config.train.ema = True
    # A large num_epochs is intentionally set here. Training will be manually stopped once sufficient
    config.save_freq = 60
    config.eval_freq = 60
    config.save_dir = 'logs/ocr/sd3_self_8gpu_1024'
    config.reward_fn = {
        "ocr": 1.0,
    }

    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config


def general_ocr_sd3_1gpu():
    gpu_number = 1
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/ocr")

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.resolution = 512
    config.sample.train_batch_size = 8
    config.sample.num_image_per_prompt = 8
    config.sample.num_batches_per_epoch = int(8/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number!"
    config.sample.test_batch_size = 16

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    # kl loss
    config.train.beta = 0.04
    # Whether to use the std of all samples or the current group's.
    config.sample.global_std = True
    config.sample.same_latent = False
    config.train.ema = True
    config.save_freq = 60
    config.eval_freq = 60
    config.save_dir = 'logs/ocr/sd3_self_1gpu'
    config.reward_fn = {
        "ocr": 1.0,
    }

    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config


def general_ocr_sd3_L_8gpu():
    """SD3.5-Large on 8 GPUs."""
    gpu_number = 8
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/ocr")

    # sd3.5 large
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-large"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 7.0

    config.resolution = 512
    config.sample.train_batch_size = 4
    config.sample.num_image_per_prompt = 16
    config.sample.num_batches_per_epoch = 8
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number!"
    config.sample.test_batch_size = 16

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0.04
    config.sample.global_std = True
    config.sample.same_latent = False
    config.train.ema = True
    config.save_freq = 60
    config.eval_freq = 60
    config.save_dir = 'logs/ocr/sd3_L_self_8gpu'
    config.reward_fn = {
        "ocr": 1.0,
    }

    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config


# ===================== Flux Configs =====================

def general_ocr_flux_8gpu():
    gpu_number=8
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/ocr")

    # flux
    config.pretrained.model = "black-forest-labs/FLUX.1-dev"
    config.sample.num_steps = 6
    config.sample.eval_num_steps = 28
    config.sample.guidance_scale = 3.5

    config.resolution = 512
    config.sample.train_batch_size = 4
    config.sample.num_image_per_prompt = 16
    config.sample.num_batches_per_epoch = 8
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number!"
    config.sample.test_batch_size = 16

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0.04
    config.sample.global_std = True
    config.sample.same_latent = False
    config.train.ema = True
    config.sample.noise_level = 0.7
    config.mixed_precision = "bf16"
    config.save_freq = 30
    config.eval_freq = 30
    config.save_dir = 'logs/ocr/flux_self_8gpu'
    config.reward_fn = {
        "ocr": 1.0,
    }

    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config


# ===================== SDXL Configs =====================

def sdxl_self_8gpu():
    """SDXL GRPO + self-confidence on 8 GPUs, 1024x1024."""
    gpu_number = 8
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/ocr")

    # SDXL base
    config.pretrained.model = "stabilityai/stable-diffusion-xl-base-1.0"
    config.sample.num_steps = 15
    config.sample.eval_num_steps = 50
    config.sample.guidance_scale = 7.0

    config.resolution = 1024
    config.sample.train_batch_size = 8
    config.sample.num_image_per_prompt = 16
    config.sample.num_batches_per_epoch = 4
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number!"
    config.sample.test_batch_size = 16

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch // 2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    # kl loss
    config.train.beta = 0.04
    # Whether to use the std of all samples or the current group's.
    config.sample.global_std = True
    config.sample.same_latent = False
    config.train.ema = True
    config.mixed_precision = "bf16"

    # Self-confidence reward settings
    config.train.num_reward_timesteps = 8
    config.train.sds = ml_collections.ConfigDict()
    config.train.sds.k = 8           # number of antithetic probes
    config.train.sds.use_step_stride = 1
    config.train.sds.scale = 1.0
    config.train.lora_path = None

    # LoRA settings for SDXL UNet
    config.use_lora = True
    config.lora_rank = 32
    config.lora_alpha = 64
    config.lora_dropout = 0.0
    config.sdxl_lora_target_modules = [
        "to_q", "to_k", "to_v", "to_out.0",
    ]

    config.save_freq = 60
    config.eval_freq = 60
    config.save_dir = 'logs/ocr/sdxl_self_8gpu'
    config.reward_fn = {
        "ocr": 1.0,
    }
    config.prompt_fn = "general_ocr"
    config.per_prompt_stat_tracking = True
    return config


def sdxl_self_1gpu():
    """SDXL GRPO + self-confidence on 1 GPU."""
    gpu_number = 1
    config = sdxl_self_8gpu()
    config.sample.train_batch_size = 8
    config.sample.num_image_per_prompt = 8
    config.sample.num_batches_per_epoch = int(8 / (gpu_number * config.sample.train_batch_size / config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number!"
    config.sample.test_batch_size = 16

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch // 2

    config.save_dir = 'logs/ocr/sdxl_self_1gpu'
    return config


def sdxl_self_4gpu():
    """SDXL GRPO + self-confidence on 4 GPUs."""
    gpu_number = 4
    config = sdxl_self_8gpu()
    config.sample.train_batch_size = 8
    config.sample.num_image_per_prompt = 16
    config.sample.num_batches_per_epoch = int(16 / (gpu_number * config.sample.train_batch_size / config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number!"

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch // 2

    config.save_dir = 'logs/ocr/sdxl_self_4gpu'
    return config


def sdxl_pickscore_8gpu():
    """SDXL GRPO + self-confidence with PickScore eval on 8 GPUs."""
    config = sdxl_self_8gpu()
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")
    config.reward_fn = {
        "pickscore": 1.0,
    }
    config.prompt_fn = "general_ocr"
    config.save_dir = 'logs/sdxl/pickscore_self_8gpu'
    return config


def get_config(name):
    return globals()[name]()
