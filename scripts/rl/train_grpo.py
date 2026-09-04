"""
GRPO Training Script using TRL

Main entry point for GRPO training using TRL's GRPOTrainer.
Supports SwanLab logging for experiment tracking.

Usage:
    python scripts/rl/train_grpo.py --config configs/rl_config.yaml
"""

import os
import sys
import argparse
import yaml
import logging
import time
import threading
from datetime import datetime

# CRITICAL: Monkey-patch to skip PyTorch 2.6 security check (CVE-2025-32434)
# This is safe because we only load our own checkpoints
from transformers.utils import import_utils
import_utils.check_torch_load_is_safe = lambda: None


import torch
import transformers
from peft import LoraConfig, get_peft_model
from trl import GRPOConfig
from transformers import TrainerCallback

# Add project root to path BEFORE importing local modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from model.lap import LaPForCausalLM
from model.llava import conversation as conversation_lib
from tools.utils import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN

from rl.rewards import create_reward_fn
from rl.dataset_wrapper import create_trl_dataset
from rl.lap_grpo_trainer import LaPGRPOTrainer

# Setup logging (force to avoid prior configs silencing logs)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True,
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="GRPO Training with TRL")
    
    parser.add_argument("--config", type=str, default="configs/rl_config.yaml")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Override model path from config")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output directory")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Override max training steps")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--vision_pretrained", type=str,
                        default="./checkpoints/sam_vit_h_4b8939.pth")
    
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to checkpoint directory to resume training from")
    
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load config from YAML."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_swanlab(config: dict, output_dir: str):
    """Initialize SwanLab for experiment tracking."""
    if not config['training'].get('use_swanlab', False):
        return None
    
    try:
        import swanlab
        
        project = config['training'].get('swanlab_project', 'LaP-Forensic-GRPO')
        experiment = config['training'].get('swanlab_experiment')
        if experiment is None:
            experiment = f"grpo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        swanlab.init(
            project=project,
            experiment_name=experiment,
            config={
                'model_path': config['model']['model_path'],
                'lora_r': config['model'].get('lora_r', 8),
                'learning_rate': config['training'].get('learning_rate', 1e-5),
                'batch_size': config['training'].get('per_device_train_batch_size', 2),
                'gradient_accumulation_steps': config['training'].get('gradient_accumulation_steps', 4),
                'num_generations': config['grpo'].get('num_generations', 4),
                'temperature': config['grpo'].get('temperature', 0.8),
                'mask_reward_weight': config['rewards'].get('mask_reward_weight', 1.0),
                'format_reward_weight': config['rewards'].get('format_reward_weight', 0.3),
            }
        )
        logger.info(f"SwanLab initialized: project={project}, experiment={experiment}")
        return swanlab
    except ImportError:
        logger.warning("SwanLab not installed. Run: pip install swanlab")
        return None
    except Exception as e:
        logger.warning(f"SwanLab initialization failed: {e}")
        return None


def setup_tokenizer(model_path: str, model_max_length: int = 1536):
    """Initialize tokenizer with special tokens."""
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path,
        model_max_length=model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    
    # Add special tokens
    special_tokens = [
        DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN,
        '<bbox>', '<point>', '[SEG]', '<p>', '</p>'
    ]
    existing = set(tokenizer.get_vocab().keys())
    to_add = [t for t in special_tokens if t not in existing]
    if to_add:
        tokenizer.add_tokens(to_add, special_tokens=True)
    
    return tokenizer


def _log_gpu_mem(label):
    """Log GPU memory usage at a checkpoint."""
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info(f"[GPU MEM] {label}: allocated={alloc:.2f} GB, reserved={reserved:.2f} GB")
        print(f"[GPU MEM] {label}: allocated={alloc:.2f} GB, reserved={reserved:.2f} GB", flush=True)

def setup_model(config: dict, tokenizer, device):
    """
    Initialize model with LoRA for RL training.
    
    Key: train_mask_decoder=False to freeze SAM decoder.
    """
    _log_gpu_mem("Before model loading")
    model_path = config['model']['model_path']
    logger.info(f"Loading model from: {model_path}")
    
    # Model args - train_mask_decoder=False for RL
    model_args = {
        'train_mask_decoder': config['model'].get('train_mask_decoder', False),  # FROZEN
        'out_dim': config['model'].get('out_dim', 256),
        'ce_loss_weight': 1.0,
        'dice_loss_weight': 0.0,
        'bce_loss_weight': 0.0,
        'scama_loss_weight': 0.0,
        'seg_token_idx': tokenizer.convert_tokens_to_ids("[SEG]"),
        'vision_pretrained': config['model'].get('vision_pretrained', './checkpoints/sam_vit_h_4b8939.pth'),
        'vision_tower': config['model'].get('vision_tower', 'openai/clip-vit-large-patch14-336'),
        'use_mm_start_end': True,
        'mm_vision_select_layer': -2,
        'with_region': True,
        'bbox_token_idx': tokenizer.convert_tokens_to_ids("<bbox>"),
        'eop_token_idx': tokenizer.convert_tokens_to_ids("</p>"),
        'bop_token_idx': tokenizer.convert_tokens_to_ids("<p>"),
        # CRITICAL: Dual-Stream Consistency settings (must match SFT training)
        'use_consistency_stream': config['model'].get('use_consistency_stream', True),
        'vae_path': config['model'].get('vae_path', 'stabilityai/sd-vae-ft-mse'),
        'consistency_encoder_type': config['model'].get('consistency_encoder_type', 'clip'),
    }
    logger.info(f"Model args: use_consistency_stream={model_args['use_consistency_stream']}")
    
    # Load model
    # CRITICAL: Set low_cpu_mem_usage=False to avoid meta tensors
    # Meta tensors cause SAM and text_hidden_fcs weights to not load properly
    # 
    # Flash Attention 2: Significantly faster attention computation
    # Requires flash-attn>=2.0 and CUDA compute capability >= 8.0 (RTX 3090+)
    use_flash_attn = os.getenv("LAP_FORENSIC_USE_FLASH_ATTN", "1") == "1"
    attn_implementation = "flash_attention_2" if use_flash_attn else "sdpa"
    logger.info(f"Using attention implementation: {attn_implementation}")
    
    model = LaPForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,  # Disable to avoid meta tensor issues
        attn_implementation=attn_implementation,  # Flash Attention 2 for speed
        **model_args
    )
    
    _log_gpu_mem("After from_pretrained (model on CPU)")
    
    # Log model parameter dtypes to check for fp32 components
    dtype_counts = {}
    for name, p in model.named_parameters():
        dt = str(p.dtype)
        dtype_counts[dt] = dtype_counts.get(dt, 0) + p.numel()
    for dt, count in dtype_counts.items():
        logger.info(f"  Params dtype {dt}: {count:,} params = {count * (4 if '32' in dt else 2) / 1024**3:.2f} GB")
    print(f"[DTYPE CHECK] {dtype_counts}", flush=True)
    
    # Configure tokens
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # Enable gradients
    model.enable_input_require_grads()
    
    # Initialize vision modules
    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch.bfloat16, device=device)
    _log_gpu_mem("After vision tower to GPU")
    
    # Freeze vision components (already done in model init, but ensure)
    for p in vision_tower.parameters():
        p.requires_grad = False
    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = False
    
    # CRITICAL: Fix SAM meta tensor issue
    # The SFT checkpoint contains trained SAM weights (mask_decoder was fine-tuned)
    # We need to load these weights, not the original SAM weights
    logger.info("Loading SAM weights from SFT checkpoint...")
    
    # Import SAM builder to create fresh SAM structure
    from model.SAM import build_sam_vit_h
    from safetensors import safe_open
    
    # Build fresh SAM (with original weights as base)
    sam_checkpoint = config['model'].get('vision_pretrained', './checkpoints/sam_vit_h_4b8939.pth')
    new_sam = build_sam_vit_h(sam_checkpoint)
    new_sam = new_sam.to(dtype=torch.bfloat16, device=device)
    
    # Now load the SFT-trained weights on top (especially mask_decoder)
    # Find all safetensor files in checkpoint
    sft_ckpt_files = [
        os.path.join(model_path, f"model-0000{i}-of-00004.safetensors")
        for i in range(1, 5)
    ]
    
    sam_state_dict = {}
    for ckpt_file in sft_ckpt_files:
        if os.path.exists(ckpt_file):
            with safe_open(ckpt_file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if 'grounding_encoder' in key:
                        # Remove 'model.grounding_encoder.' prefix
                        new_key = key.replace('model.grounding_encoder.', '')
                        sam_state_dict[new_key] = f.get_tensor(key)
    
    logger.info(f"Found {len(sam_state_dict)} SAM weights in SFT checkpoint")
    
    # Load SFT weights into SAM
    missing, unexpected = new_sam.load_state_dict(sam_state_dict, strict=False)
    if missing:
        logger.warning(f"SAM missing keys: {len(missing)} (expected for some buffers)")
    if unexpected:
        logger.warning(f"SAM unexpected keys: {len(unexpected)}")
    
    # Replace the grounding_encoder
    model.model.grounding_encoder = new_sam
    logger.info("SAM loaded from SFT checkpoint successfully")
    
    # Verify
    sample_param = next(model.model.grounding_encoder.parameters())
    logger.info(f"SAM param device: {sample_param.device}, dtype: {sample_param.dtype}, is_meta: {sample_param.is_meta}")
    
    # SAM is frozen for RL
    for p in model.model.grounding_encoder.parameters():
        p.requires_grad = False
    logger.info("Grounding encoder (SAM) frozen for RL")
    _log_gpu_mem("After SAM loaded to GPU")
    
    # CRITICAL: Check and fix text_hidden_fcs (projection layer from LLM to SAM)
    # This layer is trained during SFT and must be loaded correctly
    # Try different paths to find text_hidden_fcs
    text_hidden_fcs = None
    if hasattr(model, 'model') and hasattr(model.model, 'text_hidden_fcs'):
        text_hidden_fcs = model.model.text_hidden_fcs
        logger.info("Found text_hidden_fcs at model.model.text_hidden_fcs")
    elif hasattr(model, 'get_model') and hasattr(model.get_model(), 'text_hidden_fcs'):
        text_hidden_fcs = model.get_model().text_hidden_fcs
        logger.info("Found text_hidden_fcs at model.get_model().text_hidden_fcs")
    else:
        logger.warning("Could not find text_hidden_fcs in model!")
        # List available attributes for debugging
        if hasattr(model, 'model'):
            logger.info(f"model.model attributes: {[a for a in dir(model.model) if not a.startswith('_')][:20]}")
    
    if text_hidden_fcs is not None:
        sample_fc_param = next(text_hidden_fcs.parameters())
        logger.info(f"text_hidden_fcs param: device={sample_fc_param.device}, dtype={sample_fc_param.dtype}, is_meta={sample_fc_param.is_meta}")
        
        # Check actual weight values to verify they're not random
        weight_sample = sample_fc_param.data.flatten()[:5]
        logger.info(f"text_hidden_fcs weight sample: {weight_sample.tolist()}")
        
        if sample_fc_param.is_meta:
            logger.warning("text_hidden_fcs is on meta device! Reloading from checkpoint...")
            # Load the specific weights from checkpoint
            from safetensors import safe_open
            ckpt_path = os.path.join(model_path, "model-00004-of-00004.safetensors")
            if os.path.exists(ckpt_path):
                with safe_open(ckpt_path, framework="pt", device="cpu") as f:
                    fc_state = {}
                    for key in f.keys():
                        if 'text_hidden_fcs' in key:
                            # Remove 'model.' prefix if present
                            new_key = key.replace('model.', '')
                            fc_state[new_key] = f.get_tensor(key)
                            logger.info(f"Loaded {key} -> {new_key}")
                    
                    # Load into model
                    text_hidden_fcs.load_state_dict(fc_state, strict=False, assign=True)
                    logger.info("text_hidden_fcs reloaded successfully")
                    
                    # Move to device
                    text_hidden_fcs.to(dtype=torch.bfloat16, device=device)
                    
                    # Verify
                    sample_fc_param = next(text_hidden_fcs.parameters())
                    logger.info(f"text_hidden_fcs after reload: device={sample_fc_param.device}, is_meta={sample_fc_param.is_meta}")
        else:
            # Move to correct device/dtype if needed
            text_hidden_fcs.to(dtype=torch.bfloat16, device=device)
            logger.info("text_hidden_fcs already on real device")
    
    # Setup LoRA
    lora_r = config['model'].get('lora_r', 8)
    if lora_r > 0:
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=config['model'].get('lora_alpha', 16),
            target_modules=["q_proj", "v_proj"],
            lora_dropout=config['model'].get('lora_dropout', 0.05),
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        logger.info(f"LoRA applied with r={lora_r}")
    
    # Resize embeddings
    model.resize_token_embeddings(len(tokenizer))
    
    # Ensure text_hidden_fcs is trainable
    for name, param in model.named_parameters():
        if 'text_hidden_fcs' in name:
            param.requires_grad = True
            logger.info(f"Trainable: {name}")
    
    _log_gpu_mem("After LoRA + text_hidden_fcs setup")
    
    # Count parameters
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Parameters: {trainable:,} trainable / {total:,} total ({100*trainable/total:.2f}%)")
    
    model.to(device)
    _log_gpu_mem("After model.to(device) — FINAL")
    return model


class SwanLabCallback(TrainerCallback):
    """Callback for logging metrics to SwanLab."""
    
    def __init__(self, swanlab_module, reward_fn):
        self.swanlab = swanlab_module
        self.reward_fn = reward_fn
        self.step_count = 0
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or self.swanlab is None:
            return
        
        # Log all metrics to SwanLab
        metrics_to_log = {}
        
        # Standard training metrics
        for key, value in logs.items():
            if isinstance(value, (int, float)):
                metrics_to_log[key] = value
        
        # Add custom reward statistics if available
        if hasattr(self.reward_fn, 'last_stats'):
            stats = self.reward_fn.last_stats
            metrics_to_log['reward/mean'] = stats.get('reward_mean', 0)
            metrics_to_log['reward/min'] = stats.get('reward_min', 0)
            metrics_to_log['reward/max'] = stats.get('reward_max', 0)
            metrics_to_log['reward/format_mean'] = stats.get('format_reward_mean', 0)
            metrics_to_log['reward/mask_mean'] = stats.get('mask_reward_mean', 0)
        
        if metrics_to_log:
            self.swanlab.log(metrics_to_log, step=state.global_step)
    
    def on_train_end(self, args, state, control, **kwargs):
        if self.swanlab is not None:
            self.swanlab.finish()


class LRResetCallback(TrainerCallback):
    """
    Reset learning rate scheduler after resuming from checkpoint.
    Creates a NEW cosine scheduler that decays from initial_lr to 0
    over the remaining training steps.
    """
    
    def __init__(self, initial_lr: float, max_steps: int = 3000, reset_on_resume: bool = True):
        self.initial_lr = initial_lr
        self.max_steps = max_steps
        self.reset_on_resume = reset_on_resume
        self._has_reset = False
        self._first_call = True          # Track whether this is the first on_step_begin call
        self._is_resumed = False          # Whether we actually resumed from checkpoint
    
    def on_step_begin(self, args, state, control, **kwargs):
        """Reset LR at the first step after resume (checkpoint fully loaded by now)."""
        # On the very first call, record whether we're resuming:
        # If global_step > 0 on the FIRST call, it means Trainer loaded a checkpoint.
        # On a fresh run, the first call has global_step == 0.
        if self._first_call:
            self._is_resumed = state.global_step > 0
            self._first_call = False
            if not self._is_resumed:
                logger.info("[LRResetCallback] Fresh training detected, will NOT reset scheduler")
        
        if not self._has_reset and self.reset_on_resume and self._is_resumed:
            self._has_reset = True
            
            optimizer = kwargs.get('optimizer')
            lr_scheduler = kwargs.get('lr_scheduler')
            
            if optimizer is None or lr_scheduler is None:
                logger.warning("[LRResetCallback] optimizer or lr_scheduler not available")
                return
            
            # Calculate remaining steps
            remaining_steps = self.max_steps - state.global_step
            if remaining_steps <= 0:
                remaining_steps = 1000  # fallback
            
            # Set optimizer LR to initial value
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.initial_lr
            
            # Create a new cosine scheduler that decays from initial_lr to 0
            # over the remaining steps
            from transformers import get_cosine_schedule_with_warmup
            
            new_scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=0,  # No warmup, start directly at initial_lr
                num_training_steps=remaining_steps,
            )
            
            # Replace the old scheduler's step function with the new one
            # This is a hack but works because LambdaLR schedulers are composable
            lr_scheduler.lr_lambdas = new_scheduler.lr_lambdas
            lr_scheduler.last_epoch = -1  # Will be incremented on first step() call
            lr_scheduler.base_lrs = [self.initial_lr] * len(lr_scheduler.base_lrs)
            
            # Force update the LR
            lr_scheduler.step()
            
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f"[LRResetCallback] Created new cosine scheduler:")
            logger.info(f"  - Initial LR: {self.initial_lr}")
            logger.info(f"  - Remaining steps: {remaining_steps}")
            logger.info(f"  - Current LR after reset: {current_lr}")


class StepDebugCallback(TrainerCallback):
    """Lightweight step timing/debug callback."""

    def __init__(self, log_every: int = 1):
        self.log_every = log_every
        self._step_start = None

    def on_step_begin(self, args, state, control, **kwargs):
        self._step_start = time.perf_counter()

    def on_step_end(self, args, state, control, **kwargs):
        if self._step_start is None:
            return
        if state.global_step % self.log_every == 0:
            elapsed = time.perf_counter() - self._step_start
            logger.info("Step %d end: elapsed=%.2fs", state.global_step, elapsed)





class MaskValidationCallback(TrainerCallback):
    """
    Compute mIoU/gIoU/F1 metrics on validation set every eval_steps.

    Uses the SAME pipeline as SFT's validate_model_performance:
    teacher-forcing forward pass with inference=True, letting the model
    internally extract [SEG] hidden states and generate masks via SAM.
    """

    def __init__(self, model, val_dataset, tokenizer, reward_fn,
                 eval_steps: int = 50, swanlab=None, max_val_samples: int = None):
        self.model = model
        self.val_dataset = val_dataset
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.eval_steps = eval_steps
        self.swanlab = swanlab
        self.max_val_samples = max_val_samples
        self.debug = os.getenv("LAP_FORENSIC_DEBUG", "0") != "0"

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.eval_steps == 0 and state.global_step > 0:
            metrics = self._validate(state.global_step)
            if self.swanlab:
                self.swanlab.log(metrics, step=state.global_step)
            print(f"\n[Validation Step {state.global_step}] "
                  f"mIoU={metrics.get('val/miou', 0):.4f}, "
                  f"gIoU={metrics.get('val/giou', 0):.4f}, "
                  f"F1={metrics.get('val/f1', 0):.4f}, "
                  f"samples={metrics.get('val/samples', 0)}\n", flush=True)

    def _unwrap_model(self):
        """Unwrap PEFT/DDP wrappers to get the underlying LaPForCausalLM."""
        model = self.model
        # DDP wrapper
        if hasattr(model, 'module'):
            model = model.module
        # PEFT wrapper — get base model
        if hasattr(model, 'base_model') and hasattr(model.base_model, 'model'):
            model = model.base_model.model
        return model

    def _prepare_val_batch(self, idx: int, device):
        """
        Prepare a single validation sample in the same format as
        SFT's custom_collate_fn with inference=True.

        Returns dict matching model_forward() signature, or None on failure.
        """
        from model.llava.mm_utils import tokenizer_image_token
        from tools.utils import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
        from model.llava import conversation as conversation_lib

        # Access the raw item to trigger metadata caching
        item = self.val_dataset[idx]
        metadata = self.val_dataset.get_metadata(idx)

        if not metadata:
            return None

        gt_masks = metadata.get('masks')
        grounding_enc_image = metadata.get('grounding_enc_image')
        global_enc_image = metadata.get('global_enc_image')
        label = metadata.get('label')
        resize = metadata.get('resize')
        full_conv = metadata.get('full_conversation', '')

        if gt_masks is None or grounding_enc_image is None or not full_conv:
            return None

        # Wrap <image> with start/end tokens (same as custom_collate_fn)
        if DEFAULT_IMAGE_TOKEN in full_conv and DEFAULT_IM_START_TOKEN not in full_conv:
            replace_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            full_conv = full_conv.replace(DEFAULT_IMAGE_TOKEN, replace_token)

        # Tokenize the FULL conversation (prompt + GT answer with [SEG])
        input_ids = tokenizer_image_token(full_conv, self.tokenizer, return_tensors="pt")
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to(device)
        attention_masks = input_ids.ne(self.tokenizer.pad_token_id).to(device)

        # Prepare images
        if global_enc_image.dim() == 3:
            global_enc_image = global_enc_image.unsqueeze(0)
        global_enc_image = global_enc_image.to(device, dtype=torch.bfloat16)

        if grounding_enc_image.dim() == 3:
            grounding_enc_image = grounding_enc_image.unsqueeze(0)
        grounding_enc_image = grounding_enc_image.to(device, dtype=torch.bfloat16)

        # Prepare masks
        if isinstance(gt_masks, torch.Tensor):
            gt_masks = gt_masks.float().to(device)

        return {
            'global_enc_images': global_enc_image,
            'grounding_enc_images': grounding_enc_image,
            'input_ids': input_ids,
            'attention_masks': attention_masks,
            'masks_list': [gt_masks],
            'label_list': [label],
            'resize_list': [resize],
            'offset': torch.LongTensor([0, 1]).to(device),
            'inference': True,
        }

    def _validate(self, global_step: int) -> dict:
        """
        Run validation using the same pipeline as SFT's validate_model_performance.

        Calls model.model_forward(inference=True) which:
        1. Runs teacher-forcing forward pass on full conversation
        2. Extracts hidden states at [SEG] token positions
        3. Projects through text_hidden_fcs
        4. Generates masks via SAM decoder
        5. Returns pred_masks and gt_masks
        """
        import tqdm
        from tools.utils import intersectionAndUnionGPU

        lap_model = self._unwrap_model()
        was_training = lap_model.training
        lap_model.eval()
        device = next(lap_model.parameters()).device

        # Metric accumulators (same as SFT: 2-class intersection/union for mIoU)
        total_intersection = 0.0
        total_union = 0.0
        total_giou = 0.0
        total_tp, total_fp, total_fn = 0.0, 0.0, 0.0
        total_samples = 0
        skipped = 0
        max_samples = len(self.val_dataset) if self.max_val_samples is None else min(len(self.val_dataset), self.max_val_samples)

        for idx in tqdm.tqdm(range(max_samples), desc=f"Validation (step {global_step})"):
            try:
                batch = self._prepare_val_batch(idx, device)
                if batch is None:
                    skipped += 1
                    continue

                with torch.no_grad():
                    results = lap_model.model_forward(**batch)

                if results is None or 'pred_masks' not in results:
                    skipped += 1
                    continue

                pred_masks = results['pred_masks']
                gt_masks_out = results['gt_masks']

                if not pred_masks or gt_masks_out is None or len(gt_masks_out) == 0:
                    skipped += 1
                    continue

                gt_masks_batch = gt_masks_out[0].int().to(device)
                pred_masks_batch = (pred_masks[0] > 0).int().to(device)

                # Per-mask metrics (same as SFT validate_model_performance)
                batch_giou = 0.0
                batch_intersection = 0.0
                batch_union = 0.0
                n_masks = min(gt_masks_batch.shape[0], pred_masks_batch.shape[0])
                if n_masks == 0:
                    skipped += 1
                    continue

                for target, prediction in zip(gt_masks_batch[:n_masks], pred_masks_batch[:n_masks]):
                    intersect, union_, _ = intersectionAndUnionGPU(
                        prediction.contiguous().clone(), target.contiguous(), 2, ignore_index=255
                    )
                    batch_intersection += intersect.cpu()
                    batch_union += union_.cpu()
                    # gIoU per mask
                    iou_per_class = intersect / (union_ + 1e-5)
                    iou_per_class[union_ == 0] += 1.0
                    batch_giou += iou_per_class[1].item()

                    # F1 components (foreground class)
                    pred_fg = (prediction == 1).float()
                    target_fg = (target == 1).float()
                    total_tp += (pred_fg * target_fg).sum().item()
                    total_fp += (pred_fg * (1 - target_fg)).sum().item()
                    total_fn += ((1 - pred_fg) * target_fg).sum().item()

                total_intersection += batch_intersection
                total_union += batch_union
                total_giou += batch_giou / n_masks
                total_samples += 1

            except Exception as e:
                skipped += 1
                if self.debug:
                    import traceback
                    logger.warning(f"Validation sample {idx} failed: {e}")
                    traceback.print_exc()
            finally:
                torch.cuda.empty_cache()

        if was_training:
            lap_model.train()

        if total_samples == 0:
            logger.warning(f"Validation: 0 valid samples (skipped={skipped})")
            return {'val/miou': 0.0, 'val/giou': 0.0, 'val/f1': 0.0, 'val/samples': 0}

        # Compute metrics (same formula as SFT)
        iou_per_class = total_intersection / (total_union + 1e-10)
        class_iou = iou_per_class[1].item()       # Foreground IoU
        background_iou = iou_per_class[0].item()   # Background IoU
        mean_iou = (background_iou + class_iou) / 2.0
        global_iou = total_giou / total_samples

        precision = total_tp / (total_tp + total_fp + 1e-10)
        recall = total_tp / (total_tp + total_fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)

        logger.info(f"Validation: samples={total_samples}, skipped={skipped}, "
                     f"mIoU={mean_iou:.4f}, gIoU={global_iou:.4f}, cIoU={class_iou:.4f}, F1={f1:.4f}")

        return {
            'val/miou': mean_iou, 'val/giou': global_iou, 'val/ciou': class_iou,
            'val/f1': f1, 'val/precision': precision, 'val/recall': recall,
            'val/samples': total_samples,
        }


def main():
    args = parse_args()

    def log_print(message: str) -> None:
        logger.info(message)
        print(message, flush=True)
    
    # Load config
    config = load_config(args.config)

    # Debug env flags
    debug_flag = os.getenv("LAP_FORENSIC_DEBUG", "0")
    debug_every = os.getenv("LAP_FORENSIC_DEBUG_EVERY", "10")
    log_print(f"Debug flags: LAP_FORENSIC_DEBUG={debug_flag} LAP_FORENSIC_DEBUG_EVERY={debug_every}")

    # Periodic heartbeat to prove the process is alive
    if debug_flag != "0":
        def _heartbeat():
            start = time.time()
            while True:
                elapsed = time.time() - start
                log_print(f"Heartbeat: running for {elapsed:.0f}s")
                time.sleep(60)
        threading.Thread(target=_heartbeat, daemon=True).start()
    
    # Override with CLI args
    if args.model_path:
        config['model']['model_path'] = args.model_path
    if args.output_dir:
        config['training']['output_dir'] = args.output_dir
    # Only override max_steps if explicitly set to a positive value
    # -1 means "use config file value" or "auto-calculate from epochs"
    if args.max_steps is not None and args.max_steps > 0:
        config['training']['max_steps'] = args.max_steps
    if args.vision_pretrained:
        config['model']['vision_pretrained'] = args.vision_pretrained
    
    # Setup output dir
    output_dir = config['training'].get('output_dir', './output/rl_grpo')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(output_dir, f"grpo_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    
    logger.info(f"Output: {output_dir}")
    
    # Initialize SwanLab
    swanlab = setup_swanlab(config, output_dir)
    
    # Device
    device = torch.device(f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu")
    
    # Tokenizer
    log_print("Initializing tokenizer...")
    tokenizer = setup_tokenizer(
        config['model']['model_path'],
        config['model'].get('model_max_length', 1536),
    )
    log_print("Tokenizer ready.")
    
    # Conversation template
    conversation_lib.default_conversation = conversation_lib.conv_templates["llava_v1"]
    
    # Model
    log_print("Initializing model...")
    model = setup_model(config, tokenizer, device)
    # Enable gradient checkpointing to save VRAM (trades compute for memory)
    # TRL handles the use_cache conflict: logps forward uses use_cache=False,
    # generation (in torch.no_grad) uses use_cache=True separately.
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        log_print("Gradient checkpointing: ENABLED (saves ~50% activation memory)")
    log_print("Model ready.")
    
    # Dataset (MANDATORY: use_dual_stream must match SFT training)
    use_dual_stream = config['model'].get('use_consistency_stream', True)
    log_print(f"Creating dataset with dual-stream={use_dual_stream}...")
    train_dataset = create_trl_dataset(
        dataset_dir=config['data'].get('dataset_dir', './data'),
        tokenizer=tokenizer,
        global_image_encoder=config['model'].get('vision_tower', 'openai/clip-vit-large-patch14-336'),
        epoch_samples=config['training'].get('epoch_samples', 8000),
        precision=config['training'].get('precision', 'bf16'),
        image_size=config['data'].get('image_size', 512),
        num_classes_per_sample=config['data'].get('num_classes_per_sample', 3),
        validation=False,
        use_dual_stream=use_dual_stream,  # MANDATORY: Must match SFT
    )
    log_print("Training dataset ready.")
    
    # Validation dataset for mIoU evaluation
    # Use eval_dataset_dir if specified, otherwise fall back to training dataset_dir
    eval_dataset_dir = config['data'].get('eval_dataset_dir', config['data'].get('dataset_dir', './data'))
    log_print(f"Creating validation dataset from {eval_dataset_dir}...")
    val_dataset = create_trl_dataset(
        dataset_dir=eval_dataset_dir,
        tokenizer=tokenizer,
        global_image_encoder=config['model'].get('vision_tower', 'openai/clip-vit-large-patch14-336'),
        epoch_samples=config['training'].get('val_samples', 200),  # Smaller for validation
        precision=config['training'].get('precision', 'bf16'),
        image_size=config['data'].get('image_size', 512),
        num_classes_per_sample=config['data'].get('num_classes_per_sample', 3),
        validation=True,  # CRITICAL: Use validation split (loads test.json + test/images)
        use_dual_stream=use_dual_stream,
    )
    log_print("Validation dataset ready.")
    
    # Reward function (pass dataset for mask IoU computation)
    log_print("Initializing reward function...")
    reward_fn = create_reward_fn(
        model=model,
        tokenizer=tokenizer,
        dataset=train_dataset,  # CRITICAL: enables mask reward via idx lookup
        mask_reward_weight=config['rewards'].get('mask_reward_weight', 1.0),
        format_reward_weight=config['rewards'].get('format_reward_weight', 0.3),
        missing_seg_penalty=config['rewards'].get('missing_seg_penalty', -2.0),
    )
    log_print("Reward function ready.")
    
    # TRL GRPO Config
    grpo_config = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=config['training'].get('num_train_epochs', 3),
        max_steps=config['training'].get('max_steps', -1),
        per_device_train_batch_size=config['training'].get('per_device_train_batch_size', 4),
        gradient_accumulation_steps=config['training'].get('gradient_accumulation_steps', 4),
        learning_rate=config['training'].get('learning_rate', 1e-5),
        warmup_steps=config['training'].get('warmup_steps', 0),
        warmup_ratio=config['training'].get('warmup_ratio', 0.03),
        weight_decay=config['training'].get('weight_decay', 0.0),
        max_grad_norm=config['training'].get('max_grad_norm', 1.0),  # Gradient clipping
        lr_scheduler_type=config['training'].get('lr_scheduler_type', 'cosine'),
        logging_steps=config['training'].get('logging_steps', 10),
        save_steps=config['training'].get('save_steps', 200),
        # GRPO specific
        num_generations=config['grpo'].get('num_generations', 4),
        temperature=config['grpo'].get('temperature', 0.8),
        max_completion_length=config['grpo'].get('max_new_tokens', 512),
        beta=config['grpo'].get('beta', 0.04),  # KL divergence penalty
        # Precision & Memory
        bf16=True,
        gradient_checkpointing=False,  # Disabled: 48GB has enough room, saves 30-50% compute
        deepspeed=config['training'].get('deepspeed_config', None),
        # CRITICAL: steps_per_generation defaults to gradient_accumulation_steps (=8)
        # which batches too many multimodal samples into ONE generation call.
        # generation_batch_size = batch_size * steps_per_generation must be divisible by num_generations
        # With batch_size=4, num_generations=4: steps_per_generation=1 → generation_batch_size=4
        steps_per_generation=1,
        # Additional
        remove_unused_columns=False,
        report_to="none",  # We use SwanLab callback instead
        ddp_find_unused_parameters=False,
        # KV cache is now properly supported after fixing DynamicCache handling
        # in prepare_inputs_labels_for_multimodal
        generation_kwargs={
            "use_cache": True,
            "repetition_penalty": config['grpo'].get('repetition_penalty', 1.0),
        },
    )
    
    # Gradient checkpointing requires model.config.use_cache = False
    # TRL handles cache per-phase: use_cache=False in logps forward, use_cache=True in generation_config
    if hasattr(model, 'config'):
        model.config.use_cache = False  # Required for gradient checkpointing compatibility
        logger.info("model.config.use_cache=False (gradient checkpointing enabled; TRL manages cache per-phase)")
    
    # Log generation settings for debugging
    logger.info(f"GRPO Settings: batch_size={grpo_config.per_device_train_batch_size}, "
                f"num_generations={grpo_config.num_generations}, "
                f"max_completion_length={grpo_config.max_completion_length}, "
                f"beta={grpo_config.beta}")
    logger.info(f"Total generations per step: {grpo_config.per_device_train_batch_size * grpo_config.num_generations}")
    
    log_print("Initializing TRL GRPOTrainer...")
    
    # Callbacks
    callbacks = [StepDebugCallback(log_every=1)]
    
    # LR Reset callback - force LR to start from initial value on resume
    # Creates new cosine scheduler: initial_lr → 0 over remaining steps
    initial_lr = config['training'].get('learning_rate', 6e-6)
    max_steps = config['training'].get('max_steps', 3000)
    callbacks.append(LRResetCallback(initial_lr=initial_lr, max_steps=max_steps, reset_on_resume=True))
    log_print(f"LRResetCallback registered (initial_lr={initial_lr}, max_steps={max_steps})")
    
    if swanlab is not None:
        callbacks.append(SwanLabCallback(swanlab, reward_fn))
    
    # Mask validation callback for mIoU monitoring
    eval_steps = config['training'].get('eval_steps', 1000)
    callbacks.append(MaskValidationCallback(
        model=model,
        val_dataset=val_dataset,
        tokenizer=tokenizer,
        reward_fn=reward_fn,
        eval_steps=eval_steps,
        swanlab=swanlab,
    ))
    log_print(f"MaskValidationCallback registered (eval_steps={eval_steps})")
    
    # Create trainer (using custom LaPGRPOTrainer for multimodal support)
    trainer = LaPGRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
        callbacks=callbacks,
    )
    
    # generation_config.use_cache=True for generation (separate from model.config.use_cache)
    # model.config.use_cache stays False for gradient checkpointing compatibility
    if hasattr(trainer.model, 'generation_config'):
        trainer.model.generation_config.use_cache = True
        log_print(f"Set generation_config.use_cache = {trainer.model.generation_config.use_cache}")
    
    # Wire up completion logging (happens in training loop, no extra inference)
    log_completions_every = config['training'].get('log_completions_every', 10)
    trainer.log_completions_every = log_completions_every
    trainer._swanlab = swanlab
    log_print(f"Completion logging: every {log_completions_every} generation calls (in-loop + SwanLab)")
    
    log_print("TRL GRPOTrainer ready.")
    _log_gpu_mem("After GRPOTrainer.__init__")
    
    log_print("Starting training...")
    
    # Debug: print actual generation config before training
    if hasattr(trainer.model, 'generation_config'):
        gc = trainer.model.generation_config
        log_print(f"Final generation_config: use_cache={gc.use_cache}, max_length={getattr(gc, 'max_length', 'N/A')}")
    
    # Debug: print trainer's generation config
    if hasattr(trainer, 'generation_config'):
        tgc = trainer.generation_config
        log_print(f"Trainer generation_config: use_cache={getattr(tgc, 'use_cache', 'N/A')}, max_new_tokens={getattr(tgc, 'max_new_tokens', 'N/A')}")
    
    
    
    # Resume from checkpoint if specified
    resume_checkpoint = args.resume_from_checkpoint
    if resume_checkpoint and os.path.isdir(resume_checkpoint):
        log_print(f"Resuming training from checkpoint: {resume_checkpoint}")
    else:
        resume_checkpoint = None
    
    # CRITICAL FIX: Force _train_batch_size to match current config
    # When resuming from checkpoint, HF Trainer loads train_batch_size from trainer_state.json
    # (line ~2301 in trainer.py) which overrides our current per_device_train_batch_size.
    # If the checkpoint was saved with batch_size=8 but we now use batch_size=1,
    # the TRL dataloader creates batches of 8*32=256 samples instead of 1*32=32,
    # consuming ~32 GB just for batch data and causing OOM on 48GB GPUs.
    trainer._train_batch_size = grpo_config.per_device_train_batch_size
    log_print(f"Forced _train_batch_size = {trainer._train_batch_size} (prevents checkpoint override)")
    
    _log_gpu_mem("Before trainer.train()")
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    
    # Save final model
    logger.info("Saving final model...")
    trainer.save_model(os.path.join(output_dir, "final"))
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
