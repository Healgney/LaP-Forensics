import os
import sys
import time
import tqdm
from datetime import datetime
import random
import torch
import argparse
import deepspeed
import numpy as np
import transformers
from functools import partial
from torch.utils.data import ConcatDataset
from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from model.lap import LaPForCausalLM
from model.llava import conversation as conversation_lib

from dataset.dataset import custom_collate_fn, HybridSegDataset, HybridRegDataset, HybridCapDataset
from tools.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, AverageMeter, ProgressMeter, dict_to_cuda,
                         Summary, intersectionAndUnionGPU)

from dataset.gcg_datasets.GranDf_gcg_ds import LaPGCGDataset
import wandb

TOTAL_data_num = 0


class Logger:
    """Logger that writes to both stdout and a file."""
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log_file = log_file
        # Create directory if not exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        self.log = open(log_file, 'a', encoding='utf-8')
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # Ensure immediate write
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()
        
    def close(self):
        self.log.close()


def setup_logging(log_dir, exp_name, local_rank):
    """Setup logging to file and stdout."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f"{exp_name}_rank{local_rank}_{timestamp}.log"
    log_path = os.path.join(log_dir, log_filename)
    
    # Redirect stdout and stderr to logger
    logger = Logger(log_path)
    sys.stdout = logger
    sys.stderr = logger
    
    print(f"\n{'='*60}")
    print(f"Logging started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Log file: {log_path}")
    print(f"{'='*60}\n")
    
    return logger


def parse_args(args):
    parser = argparse.ArgumentParser(description="LaP-Forensic Localization & EXplanation Training")

    # Model-specific settings
    parser.add_argument("--version", default="")
    parser.add_argument("--vision_pretrained", default="./checkpoints/sam_vit_h_4b8939.pth", type=str)
    parser.add_argument("--vision-tower", default="openai/clip-vit-large-patch14-336", type=str)
    parser.add_argument("--conv_type", default="llava_v1", type=str, choices=["llava_v1", "llava_llama_2"])
    parser.add_argument("--tune_mm_mlp_adapter", action="store_true")
    parser.add_argument("--freeze_mm_mlp_adapter", action="store_true")
    parser.add_argument("--mm_use_im_start_end", action="store_true", default=True)
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--image_size", default=512, type=int, help="Image size for grounding image encoder")
    parser.add_argument("--model_max_length", default=1536, type=int)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj", type=str)
    parser.add_argument("--with_region", action="store_true", default=True)
    parser.add_argument("--mm_vision_select_layer", default=-2, type=int)
    parser.add_argument("--pretrain_mm_mlp_adapter", default="", type=str)
    parser.add_argument("--precision", default='bf16', type=str)

    # Dataset settings
    parser.add_argument("--use_cap_data", action="store_true", help="Use caption data")
    parser.add_argument("--use_reg_data", action="store_true", help="Use region data")
    parser.add_argument("--use_segm_data", action="store_true", help="Use segmentation data")
    parser.add_argument("--weight_cap", default=0.15, type=float, help="Sampling weight for caption data")
    parser.add_argument("--weight_reg", default=0.40, type=float, help="Sampling weight for region data")
    parser.add_argument("--weight_segm", default=0.45, type=float, help="Sampling weight for segmentation data")
    parser.add_argument("--dataset_dir", default="./data", type=str)
    parser.add_argument("--seg_dataset", default="Semantic_Segm||Refer_Segm||RefCoco_GCG||PSG_GCG||Flickr_GCG||GranDf_GCG",
                        type=str, help="Choose from: Semantic_Segm, Refer_Segm, RefCoco_GCG, GranDf_GCG, PSG_GCG, Flickr_GCG, GrandRefer_Segm")
    parser.add_argument("--segm_sample_rates", default="5,4,3,3,3,1", type=str)
    parser.add_argument("--reg_dataset", default="RefCoco_Reg||RefCocoG_Reg||RefCocoP_Reg||VisGen_Reg",
                        type=str, help="Choose from: RefCoco_Reg, RefCocoG_Reg, RefCocoP_Reg, VisGen_Reg, Flickr_Reg, GrandRefer_Reg")
    parser.add_argument("--reg_sample_rates", default="1,1,1,1", type=str)
    parser.add_argument("--cap_dataset", default="CocoCap||LLaVaInstruct", type=str,
                        help="Choose from: CocoCap, LLaVaInstruct, GrandCaptionDataset")
    parser.add_argument("--cap_sample_rates", default="1,1", type=str)
    parser.add_argument("--semantic_segm_data", default="ade20k||cocostuff||pascal_part||paco_lvis||mapillary", type=str)
    parser.add_argument("--refer_segm_data", default="refcoco||refcoco+||refcocog||refclef", type=str)
    parser.add_argument("--vqa_data", default="llava_instruct_150k", type=str)
    parser.add_argument("--num_classes_per_sample", default=3, type=int)

    # Training settings
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--auto_resume", action="store_true")
    parser.add_argument("--weight", default="", type=str)
    parser.add_argument("--lr", default=0.0003, type=float)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--steps_per_epoch", default=500, type=int)
    parser.add_argument("--batch_size", default=2, type=int, help="batch size per device per step")
    parser.add_argument("--grad_accumulation_steps", default=1, type=int)
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--workers", default=2, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--dice_loss_weight", default=1.0, type=float)
    parser.add_argument("--bce_loss_weight", default=1.0, type=float)
    parser.add_argument("--scama_loss_weight", default=0.0, type=float, help="Weight for SCAMA loss (attention-mask alignment)")
    
    # Dual-Stream Consistency settings
    parser.add_argument("--use_consistency_stream", action="store_true", help="Enable dual-stream with VAE consistency features")
    parser.add_argument("--vae_path", default="stabilityai/sd-vae-ft-mse", type=str, help="Path or HuggingFace ID for VAE model")
    parser.add_argument("--consistency_encoder_type", default="clip", choices=["clip", "cnn"], help="Encoder type for residual map features")
    
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.95, type=float)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--train_mask_decoder", action="store_true", default=True)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--epoch_samples", default=8000, type=int)
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")


    # Evaluation settings
    parser.add_argument("--val_dataset", default="CocoCapVal|RefCOCOgRegVal|RefCOCOgSegmVal", type=str,
                        help="Choose from: CocoCapVal, RefCOCOgRegVal, VisGenomeRegVal, RefCOCOgSegmVal, PsgGCGVal, "
                             "RefCocoGCGVal, FlickrGCGVal")
    parser.add_argument("--mask_validation", action="store_true")
    parser.add_argument("--no_eval", action="store_true")
    parser.add_argument("--eval_only", action="store_true")

    # Experiment settings
    parser.add_argument("--log_base_dir", default="./output_origin_multi_prompt", type=str)
    parser.add_argument("--exp_name", default="GlamFinetuneOS", type=str)
    parser.add_argument('--wandb_log',action='store_true', help='Enable Weights & Biases logging')
    return parser.parse_args(args)


def initialize_environment(args):
    """ Set up logging and model directories. """
    args.log_dir = os.path.join(args.log_base_dir, args.exp_name)
    if args.local_rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        return SummaryWriter(args.log_dir)
    return None


def setup_tokenizer_and_special_tokens(args):
    """ Load tokenizer and add special tokens. """
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version, model_max_length=args.model_max_length, padding_side="right", use_fast=False
    )
    print('\033[92m' + "---- Initialized tokenizer from: {} ----".format(args.version) + '\033[0m')
    tokenizer.pad_token = tokenizer.unk_token

    if not args.pretrained:
        if args.use_mm_start_end:
            tokenizer.add_tokens(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
            )
        # modifications specific for regions
        reg_tokens = ['<bbox>', '<point>']
        # Adding special tokens for pixel grounding
        segmentation_tokens = ['[SEG]']
        # Adding tokens for GCG
        phrase_tokens = ['<p>', '</p>']
        special_tokens = reg_tokens + segmentation_tokens + phrase_tokens
        tokenizer.add_tokens(special_tokens, special_tokens=True)

    args.bbox_token_idx = tokenizer.convert_tokens_to_ids("<bbox>")
    args.seg_token_idx = tokenizer.convert_tokens_to_ids("[SEG]")
    args.bop_token_idx = tokenizer.convert_tokens_to_ids("<p>")
    args.eop_token_idx = tokenizer.convert_tokens_to_ids("</p>")

    return tokenizer


def initialize_model(args, tokenizer):
    model_args = {k: getattr(args, k) for k in
                  ["train_mask_decoder", "out_dim", "ce_loss_weight", "dice_loss_weight", "bce_loss_weight",
                   "scama_loss_weight", "seg_token_idx", "vision_pretrained", "vision_tower", "use_mm_start_end", 
                   "mm_vision_select_layer", "pretrain_mm_mlp_adapter", "tune_mm_mlp_adapter", "freeze_mm_mlp_adapter", 
                   "mm_use_im_start_end", "with_region", "bbox_token_idx", "eop_token_idx", "bop_token_idx",
                   "use_consistency_stream", "vae_path", "consistency_encoder_type"]}  # Dual-Stream args added
    model_args["num_level_reg_features"] = 4


    # 如果是分割
    model = LaPForCausalLM.from_pretrained(
        args.version, torch_dtype=torch.bfloat16,
        #  low_cpu_mem_usage=True,
          **model_args
    )
    print('\033[92m' + "---- Initialized model from: {} ----".format(args.version) + '\033[0m')

    # Configure model tokens
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    return model


def prepare_model_for_training(model, tokenizer, args):
    # Enable input gradients
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    # Initialize vision tower
    print(
        '\033[92m' + "---- Initialized Global Image Encoder (vision tower) from: {} ----".format(
            args.vision_tower
        ) + '\033[0m'
    )
    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch.bfloat16, device=args.local_rank)

    if not args.pretrained:
        model.get_model().initialize_lap_model(model.get_model().config)
    else:
        for param in model.get_model().grounding_encoder.parameters():
            param.requires_grad = False
        if model.get_model().config.train_mask_decoder:
            model.get_model().grounding_encoder.mask_decoder.train()
            for param in model.get_model().grounding_encoder.mask_decoder.parameters():
                param.requires_grad = True

        # Projection layer
        model.get_model().text_hidden_fcs.train()
        for param in model.get_model().text_hidden_fcs.parameters():
            param.requires_grad = True

    # Set requires_grad for vision tower and mm projector
    for p in vision_tower.parameters():
        p.requires_grad = False
    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = False

    # Set requires_grad based on LoRA training
    lora_r = args.lora_r
    if lora_r == 0:
        for p in model.get_model().layers.parameters():
            p.requires_grad = True
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = True
        # Dual-Stream: Enable training for consistency projector
        if getattr(args, 'use_consistency_stream', False) and hasattr(model.get_model(), 'consistency_projector'):
            for p in model.get_model().consistency_projector.parameters():
                p.requires_grad = True

    # Configure conversation library
    conversation_lib.default_conversation = conversation_lib.conv_templates[args.conv_type]

    # Configure LoRA if applicable
    if lora_r > 0:
        lora_config = setup_lora_config(model, args)
        model = get_peft_model(model, lora_config)

    # Resize token embeddings
    model.resize_token_embeddings(len(tokenizer))

    # Make certain modules trainable
    set_trainable_modules(model)
    
    return model  # Return the (potentially LoRA-wrapped) model


def setup_lora_config(model, args):
    """ Configure LoRA settings for the model. """

    def find_proj_layers(model, target_modules):
        """ Identify projection layers in the model for LoRA adaptation. """
        linear_cls = torch.nn.Linear
        lora_module_names = set()
        for name, module in model.named_modules():
            if (isinstance(module, linear_cls) and all(
                    x not in name for x in ["grounding_encoder", "vision_tower", "mm_projector", "text_hidden_fcs"]
            ) and any(x in name for x in target_modules)):
                lora_module_names.add(name)
        return sorted(list(lora_module_names))

    # Extracting LoRA target modules
    lora_target_modules = args.lora_target_modules.split(",")
    lora_module_names = find_proj_layers(model, lora_target_modules)

    # Configuring LoRA
    lora_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, target_modules=lora_module_names, lora_dropout=args.lora_dropout,
        bias="none", task_type="CAUSAL_LM"
    )
    return lora_config


def set_trainable_modules(model):
    """ Make specified modules in the model trainable. """
    trainable_modules = [
        "lm_head",
        "embed_tokens",
        "mask_decoder",
        "text_hidden_fcs",
        "region_encoder",
        "consistency_projector",  # Dual-stream projector should remain trainable under LoRA
    ]
    for name, param in model.named_parameters():
        if any(module in name for module in trainable_modules):
            print(f"Making trainable: {name}, Shape: {param.shape}")
            param.requires_grad = True

    def count_parameters(model):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print('\033[92m' + "---- Total parameters: ----{}".format(total_params) + '\033[0m')
        print('\033[92m' + "---- Trainable parameters: ----{}".format(trainable_params) + '\033[0m')

    count_parameters(model)


def initialize_datasets_and_loaders(args, tokenizer):
    world_size = torch.cuda.device_count()
    args.distributed = world_size > 1

    # Common dataset arguments
    # Note: use_dual_stream enables consistency text injection in dataset
    use_dual_stream = getattr(args, 'use_consistency_stream', False)
    common_ds_args = {
        "dataset_dir": args.dataset_dir, 
        "tokenizer": tokenizer,
        "global_image_encoder": args.vision_tower,
        "epoch_samples": args.epoch_samples,
        "precision": args.precision, 
        "image_size": args.image_size,
        "num_classes_per_sample": args.num_classes_per_sample,
        "use_dual_stream": use_dual_stream,  # Pass dual-stream flag to datasets
    }
    
    if use_dual_stream:
        print('\033[92m' + "---- Dual-Stream Mode Enabled: Dataset will use consistency prompts ----" + '\033[0m')

    # Training datasets
    cap_train_dataset = HybridCapDataset(
        **common_ds_args, dataset=args.cap_dataset, sample_rate=[float(x) for x in args.cap_sample_rates.split(",")],
        batch_size=args.batch_size, ) if args.use_cap_data else None
    reg_train_dataset = HybridRegDataset(
        **common_ds_args, dataset=args.reg_dataset, sample_rate=[float(x) for x in args.reg_sample_rates.split(",")],
        batch_size=args.batch_size, ) if args.use_reg_data else None
    seg_train_dataset = HybridSegDataset(
        **common_ds_args, dataset=args.seg_dataset, sample_rate=[float(x) for x in args.segm_sample_rates.split(",")],
        semantic_segm_data=args.semantic_segm_data, refer_segm_data=args.refer_segm_data,
        batch_size=args.batch_size, ) if args.use_segm_data else None
 
    # Validation datasets
    val_datasets = []
    if not args.no_eval:
        val_dataset_classes = {
                               'LaP': LaPGCGDataset,
                               }
        for val_dataset_name in args.val_dataset.split('|'):
            val_dataset_class = val_dataset_classes.get(val_dataset_name)
            if val_dataset_class:
                val_datasets.append(val_dataset_class(**common_ds_args, validation=True))

    return cap_train_dataset, reg_train_dataset, seg_train_dataset, val_datasets


def setup_data_loaders(args, cap_train_dataset, reg_train_dataset, seg_train_dataset, val_datasets, tokenizer):
    # Training sampler: shuffle=True, drop_last=False (for full epoch coverage)
    train_sampler_args = {"shuffle": True, "drop_last": False}
    val_sampler_args = {"shuffle": False, "drop_last": True}
    
    train_loader_args = {"batch_size": args.batch_size, "shuffle": False, "num_workers": args.workers,
                         "pin_memory": False}
    val_loader_args = {"batch_size": args.val_batch_size, "shuffle": False, "num_workers": args.workers,
                       "pin_memory": False}
    collate_fn_args_train = partial(
        custom_collate_fn, tokenizer=tokenizer, use_mm_start_end=args.use_mm_start_end, local_rank=args.local_rank,
        inference=False
    )
    inference_mode = args.mask_validation
    collate_fn_args_val = partial(
        custom_collate_fn, tokenizer=tokenizer, use_mm_start_end=args.use_mm_start_end, local_rank=args.local_rank,
        inference=inference_mode
    )

    # Training loaders
    cap_train_loader = torch.utils.data.DataLoader(
        cap_train_dataset, sampler=torch.utils.data.distributed.DistributedSampler(
            cap_train_dataset, **train_sampler_args
        ), collate_fn=collate_fn_args_train, **train_loader_args
    ) if cap_train_dataset is not None else None
    reg_train_loader = torch.utils.data.DataLoader(
        reg_train_dataset, sampler=torch.utils.data.distributed.DistributedSampler(
            reg_train_dataset, **train_sampler_args
        ), collate_fn=collate_fn_args_train, **train_loader_args
    ) if reg_train_dataset is not None else None
    seg_train_loader = torch.utils.data.DataLoader(
        seg_train_dataset, sampler=torch.utils.data.distributed.DistributedSampler(
            seg_train_dataset, **train_sampler_args
        ), collate_fn=collate_fn_args_train, **train_loader_args
    ) if seg_train_dataset is not None else None

    # Validation loader with drop_last=True to prevent multi-GPU deadlock
    val_loader = None
    if val_datasets:
        combined_val_datasets = ConcatDataset(val_datasets)
        val_loader = torch.utils.data.DataLoader(
            combined_val_datasets, **val_loader_args, collate_fn=collate_fn_args_val,
            sampler=torch.utils.data.distributed.DistributedSampler(combined_val_datasets, **val_sampler_args), )

    return cap_train_loader, reg_train_loader, seg_train_loader, val_loader


def initialize_deepspeed(model, tokenizer, args):
    # ZeRO Stage 2 configuration for multi-GPU training
    # Stage 2 partitions: optimizer states + gradients (NOT model parameters)
    # This is more stable than Stage 3 for mixed real/fake image training because
    # Stage 3 requires parameter all-gather during forward pass which can deadlock
    # if different GPUs have different forward paths
    
    zero_optimization_config = {
    "stage": 3,
    "contiguous_gradients": True,
    "overlap_comm": True,
    "reduce_scatter": True,
    "reduce_bucket_size": 5e7,
    "allgather_bucket_size": 5e7,
    "stage3_prefetch_bucket_size": 5e7,
    "stage3_param_persistence_threshold": 1e5,
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    }
    
    
#     zero_optimization_config = {
#         "stage": 2,
#         "contiguous_gradients": True,
#         "overlap_comm": True,
#         "reduce_scatter": True,
#         "reduce_bucket_size": 5e7,
#         "allgather_bucket_size": 5e7,
#     }
    
    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.grad_accumulation_steps,
        "optimizer": {
            "type": "AdamW", 
            "params": {
                "lr": args.lr, 
                "weight_decay": 0.0,
                "betas": (args.beta1, args.beta2)
            }
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "total_num_steps": args.epochs * args.steps_per_epoch, 
                "warmup_min_lr": 0,
                "warmup_max_lr": args.lr, 
                "warmup_num_steps": 100, 
                "warmup_type": "linear"
            }
        },
        "fp16": {"enabled": args.precision == "fp16"}, 
        "bf16": {"enabled": args.precision == "bf16"},
        "gradient_clipping": 1.0,
        "zero_optimization": zero_optimization_config,
    }

    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model, model_parameters=model.parameters(), collate_fn=partial(
            custom_collate_fn, tokenizer=tokenizer, use_mm_start_end=args.use_mm_start_end, local_rank=args.local_rank
        ), config=ds_config
    )

    return model_engine, optimizer, scheduler


def resume_training_from_checkpoint(model_engine, args):
    if args.auto_resume and not args.resume:
        resume = os.path.join(args.log_dir, "ckpt_model")
        if os.path.exists(resume):
            args.resume = resume

    if args.resume:
        load_path, client_state = model_engine.load_checkpoint(args.resume)
        # with open(os.path.join(args.resume, "latest"), "r") as f:
        #     ckpt_dir = f.readlines()[0].strip()
        # args.start_epoch = int(ckpt_dir.replace("global_step", "")) // args.steps_per_epoch
        print(f"Resume training from {args.resume}, start from epoch {args.start_epoch}")


def main(args):
    print(f"[Rank {args.local_rank}] Starting main()...")
    tokenizer = setup_tokenizer_and_special_tokens(args)
    print(f"[Rank {args.local_rank}] Tokenizer initialized.")
    
    model = initialize_model(args, tokenizer)
    print(f"[Rank {args.local_rank}] Model initialized.")
    
    model = prepare_model_for_training(model, tokenizer, args)  # Capture the returned model
    print(f"[Rank {args.local_rank}] Model prepared for training.")

    model_engine, optimizer, scheduler = initialize_deepspeed(model, tokenizer, args)
    print(f"[Rank {args.local_rank}] DeepSpeed initialized.")
    
    resume_training_from_checkpoint(model_engine, args)
    print(f"[Rank {args.local_rank}] Checkpoint loaded (if any).")

    cap_train_dataset, reg_train_dataset, seg_train_dataset, val_datasets = (
        initialize_datasets_and_loaders(args, tokenizer))
    print(f"[Rank {args.local_rank}] Datasets initialized.")
    
    cap_train_loader, reg_train_loader, seg_train_loader, val_loader = (
        setup_data_loaders(args, cap_train_dataset, reg_train_dataset, seg_train_dataset, val_datasets, tokenizer))
    print(f"[Rank {args.local_rank}] DataLoaders set up.")
    # pdb.set_trace()
    # Determine active datasets and their weights
    active_dataloaders = []
    weights = []

    if args.use_cap_data:
        active_dataloaders.append(('cap', cap_train_loader))
        weights.append(args.weight_cap)
    if args.use_reg_data:
        active_dataloaders.append(('reg', reg_train_loader))
        weights.append(args.weight_reg)
    if args.use_segm_data:
        active_dataloaders.append(('seg', seg_train_loader))
        weights.append(args.weight_segm)

    # Assert that at least one dataset is active
    assert active_dataloaders, "Error: At least one dataset (segm, reg, or cap) must be active."

    print(f"[Rank {args.local_rank}] Creating dataset iterators...")
    dataset_iters = {'cap': iter(cap_train_loader) if args.use_cap_data else None,
                     'reg': iter(reg_train_loader) if args.use_reg_data else None,
                     'seg': iter(seg_train_loader) if args.use_segm_data else None, }
    print(f"[Rank {args.local_rank}] Dataset iterators created.")

    writer = initialize_environment(args)
    print(f"[Rank {args.local_rank}] TensorBoard writer initialized.")

    if args.wandb_log and args.local_rank == 0:
        run_name = f"ce_{args.ce_loss_weight}_dice_{args.dice_loss_weight}_bce_{args.bce_loss_weight}_lr_{args.lr}_epoch_{args.epochs}"
        wandb.init(
            project='LaP-Forensic',
            name=run_name,
            config=args.__dict__,
            save_code=True,
            reinit=True
        )
        # wandb.tensorboard.patch(root_logdir=args.log_base_dir)


    if args.eval_only:
        val_metrics = validate_model_performance(val_loader, model_engine, 0, writer, args)
        if args.local_rank == 0:
            print(f"Evaluation Results: {val_metrics}")
        exit()

    epoch_seeds = [random.randint(0, 100000) for _ in range(args.epochs)]
    dataset_choices = [idx for idx, _ in enumerate(active_dataloaders)]

    best_miou, best_val_loss = 0.0, np.inf
    for epoch in range(args.start_epoch, args.epochs):
        # if epoch >= 1:
        #     break

        random.seed(epoch_seeds[epoch])

        step_choices = random.choices(dataset_choices, weights=weights, k=args.steps_per_epoch)

        dataset_iters = train(
            active_dataloaders, model_engine, epoch, scheduler, writer, dataset_iters, args, step_choices, val_loader
        )

        if args.mask_validation:
            val_metrics = validate_model_performance(val_loader, model_engine, epoch, writer, args)
            giou, ciou, miou = val_metrics["giou"], val_metrics["ciou"], val_metrics["miou"]
            is_best = miou > best_miou
            best_miou = max(miou, best_miou)
            if args.local_rank == 0:  # Log the progress
                print(f"Epoch: {epoch}, giou: {giou:.4f}, ciou: {ciou:.4f}, miou: {miou:.4f}, F1: {val_metrics['f1']:.4f}, best_miou: {best_miou:.4f}")
            save_checkpoint(model_engine, args, epoch, 'miou', f"{miou:.4f}", is_best, tokenizer)
        else:
            cur_val_loss = validate_model_performance(val_loader, model_engine, epoch, writer, args)
            is_best = cur_val_loss["loss"] < best_val_loss
            best_val_loss = min(cur_val_loss["loss"], best_val_loss)
            if args.local_rank == 0:  # Log the progress
                print(f"Epoch: {epoch}, loss: {cur_val_loss['loss']:.4f}, ce_loss:{cur_val_loss['ce_loss']:.4f}, mask_bce_loss:{cur_val_loss['mask_bce_loss']:.4f}, mask_dice_loss:{cur_val_loss['mask_dice_loss']:.4f}, Best Validation Loss: {best_val_loss:.4f}")
            save_checkpoint(model_engine, args, epoch, 'loss', f"{cur_val_loss['loss']:.4f}", is_best, tokenizer)


def save_merged_model(model_engine, tokenizer, args, epoch, metric_name, metric_value):
    """
    Merge LoRA weights into the base model and save a complete HuggingFace-compatible checkpoint.
    Only called on rank 0.
    """
    from peft import PeftModel
    
    merged_dir = os.path.join(args.log_dir, f"merged_model_epoch_{epoch}_{metric_name}_{metric_value}")
    os.makedirs(merged_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Merging LoRA weights and saving to: {merged_dir}")
    print(f"{'='*60}\n")
    
    try:
        # Get the underlying model from DeepSpeed engine
        model = model_engine.module
        
        # Check if model is a PeftModel (has LoRA adapters)
        if isinstance(model, PeftModel):
            # Merge LoRA weights into base model
            print("Merging LoRA adapters into base model...")
            merged_model = model.merge_and_unload()
            print("LoRA merge complete!")
        else:
            print("Model is not a PeftModel, saving as-is...")
            merged_model = model
        
        # Save the merged model
        print("Saving merged model weights...")
        merged_model.save_pretrained(merged_dir)
        
        # Save tokenizer
        print("Saving tokenizer...")
        tokenizer.save_pretrained(merged_dir)
        
        # Save training args for reference
        import json
        args_dict = vars(args).copy()
        # Remove non-serializable items
        for key in list(args_dict.keys()):
            try:
                json.dumps(args_dict[key])
            except (TypeError, ValueError):
                del args_dict[key]
        with open(os.path.join(merged_dir, "training_args.json"), "w") as f:
            json.dump(args_dict, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"Merged model saved successfully to: {merged_dir}")
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"Error during model merging: {e}")
        import traceback
        traceback.print_exc()


def save_checkpoint(model_engine, args, epoch, metric_name, metric_value, is_best, tokenizer=None):
    """ Saves the model checkpoint. If is_best, also save merged model. """
    save_dir_name = f"ckpt_model_ce_{args.ce_loss_weight}_dice_{args.dice_loss_weight}_bce_{args.bce_loss_weight}"
    save_dir = os.path.join(args.log_dir, save_dir_name)
    # Ensure the directory exists
    if args.local_rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        ckpt_filename = f"epoch_{epoch}_val_{metric_name}_{metric_value}.pth"
        torch.save({"epoch": epoch, f"val_{metric_name}": metric_value}, os.path.join(save_dir, ckpt_filename))
    torch.distributed.barrier()
    model_engine.save_checkpoint(save_dir)
    
    # Merge and save complete model when we get the best checkpoint
    if is_best and args.local_rank == 0 and tokenizer is not None:
        save_merged_model(model_engine, tokenizer, args, epoch, metric_name, metric_value)


def train(active_datasets, model, epoch, scheduler, writer, dataset_iters, args, step_choices, val_loader):
    """Main training loop."""

    def get_next_input(iterator, data_loader):
        """Retrieve next input from the iterator, or reinitialize if necessary."""
        try:
            return next(iterator), iterator
        except StopIteration:
            new_iterator = iter(data_loader)
            return next(new_iterator), new_iterator

    def format_time(seconds):
        """Format seconds into human-readable time string."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"

    def log_progress():
        """Log training progress with time estimates."""
        # Only validate after some training steps, not at step 0
        # This avoids potential multi-GPU sync issues at the very beginning
        if global_step > 0 and global_step % 100 == 0:
            val_metrics = validate_model_performance(val_loader, model, epoch, writer, args)
            if args.wandb_log and args.local_rank == 0:
                if args.mask_validation:
                    # Log segmentation metrics
                    wandb.log({
                        'val_giou': val_metrics["giou"],
                        'val_ciou': val_metrics["ciou"],
                        'val_miou': val_metrics["miou"],
                        'val_f1': val_metrics["f1"],
                        'val_precision': val_metrics["precision"],
                        'val_recall': val_metrics["recall"],
                    }, step=global_step + args.steps_per_epoch * epoch)
                else:
                    # Log loss metrics
                    wandb.log({
                        'val_ce_loss': val_metrics["ce_loss"],
                        'val_loss': val_metrics["loss"],
                        'val_mask_bce_loss': val_metrics["mask_bce_loss"],
                        'val_mask_dice_loss': val_metrics["mask_dice_loss"],
                        'val_mask_loss': val_metrics["mask_loss"],
                        'val_scama_loss': val_metrics.get("scama_loss", 0.0),
                    }, step=global_step + args.steps_per_epoch * epoch)

        if global_step % args.print_freq == 0:
            if args.distributed:
                for tracker in trackers.values():
                    tracker.all_reduce()

            if args.local_rank == 0:
                # Calculate time statistics
                elapsed_time = time.time() - epoch_start_time
                steps_completed = global_step + 1
                steps_remaining = args.steps_per_epoch - steps_completed
                
                if steps_completed > 0:
                    time_per_step = elapsed_time / steps_completed
                    eta_epoch = time_per_step * steps_remaining
                    total_steps_remaining = steps_remaining + (args.epochs - epoch - 1) * args.steps_per_epoch
                    eta_total = time_per_step * total_steps_remaining
                else:
                    eta_epoch = 0
                    eta_total = 0
                
                # Print progress with time info
                time_info = f"[Elapsed: {format_time(elapsed_time)} | ETA(epoch): {format_time(eta_epoch)} | ETA(total): {format_time(eta_total)}]"
                progress.display(global_step + 1)
                print(f"    {time_info}")
                
                for key, tracker in trackers.items():
                    writer.add_scalar(f"train/{key}", tracker.avg, global_step)
                    if args.wandb_log and args.local_rank == 0:
                        wandb.log({f"train/{key}": tracker.avg}, step=global_step + args.steps_per_epoch * epoch)
                writer.add_scalar("metrics/total_secs_per_batch", batch_time.avg, global_step)
                writer.add_scalar("metrics/data_secs_per_batch", data_time.avg, global_step)
                if args.wandb_log and args.local_rank == 0:
                    wandb.log({
                        "metrics/total_secs_per_batch": batch_time.avg,
                        "metrics/data_secs_per_batch": data_time.avg
                    }, step=global_step + args.steps_per_epoch * epoch) 

            for tracker in trackers.values():
                tracker.reset()

    batch_time = AverageMeter("Time", ":.4f")
    data_time = AverageMeter("Data", ":.4f")
    trackers = {"loss": AverageMeter("Loss", ":.4f"),
                "ce_loss": AverageMeter("CeLoss", ":.4f"),
                "mask_bce_loss": AverageMeter("MaskBCELoss", ":.4f"),
                "mask_dice_loss": AverageMeter("MaskDICELoss", ":.4f"),
                "mask_loss": AverageMeter("MaskLoss", ":.4f"),
                "scama_loss": AverageMeter("SCAMALoss", ":.4f")}
    progress = ProgressMeter(args.steps_per_epoch, list(trackers.values()), prefix=f"Epoch: [{epoch}]")

    model.train()
    epoch_start_time = time.time()  # Track epoch start time
    end = time.time()
    print(f"[Rank {args.local_rank}] Starting training loop for epoch {epoch}...")
    for global_step in range(args.steps_per_epoch):
        if global_step == 0:
            print(f"[Rank {args.local_rank}] Entering step 0...")
        for _ in range(args.grad_accumulation_steps):
            # Select data loader based on step choice
            dataset_type, data_loader = active_datasets[step_choices[global_step]]
            if global_step == 0:
                print(f"[Rank {args.local_rank}] Getting batch from {dataset_type}...")
            data_batch, new_iter = get_next_input(dataset_iters[dataset_type], data_loader)
            dataset_iters[dataset_type] = new_iter
            if global_step == 0:
                print(f"[Rank {args.local_rank}] Batch loaded, moving to CUDA...")

            data_time.update(time.time() - end)
            # Prepare data and convert relevant tensors to bfloat16
            data_batch = dict_to_cuda(data_batch)
            for key in ["global_enc_images", "grounding_enc_images"]:
                if data_batch[key] is not None:
                    data_batch[key] = data_batch[key].bfloat16()
            
            if global_step == 0:
                print(f"[Rank {args.local_rank}] Starting forward pass...", flush=True)

            output_dict = model(**data_batch)
            
            if global_step == 0:
                print(f"[Rank {args.local_rank}] Forward pass done, loss={output_dict['loss'].item():.4f}", flush=True)

            # Update training metrics
            for key, tracker in trackers.items():
                if key in output_dict:
                    tracker.update(output_dict[key].item(), data_batch["global_enc_images"].size(0))

            if global_step == 0:
                print(f"[Rank {args.local_rank}] Starting backward pass...", flush=True)
            model.backward(output_dict["loss"])
            if global_step == 0:
                print(f"[Rank {args.local_rank}] Backward done, stepping optimizer...", flush=True)
            model.step()
            if global_step == 0:
                print(f"[Rank {args.local_rank}] Step 0 completed!", flush=True)

        batch_time.update(time.time() - end)
        end = time.time()
        log_progress()

        if global_step != 0:
            curr_lr = scheduler.get_last_lr()
            if args.local_rank == 0:
                writer.add_scalar("train/lr", curr_lr[0], global_step)

    return dataset_iters


def validate_model_performance(validation_loader, training_model, current_epoch, tensorboard_writer, args):
    # Synchronize all GPUs before starting validation to prevent deadlock
    # if args.distributed:
    #     torch.distributed.barrier()
    
    if args.mask_validation:
        # For use with only segmentation/GCG type datasets
        trackers = {"intersection": AverageMeter("Intersec", ":.4f", Summary.SUM),
                    "union": AverageMeter("Union", ":.4f", Summary.SUM),
                    "gIoU": AverageMeter("gIoU", ":.4f", Summary.SUM),
                    # F1 score components (pixel-level)
                    "TP": AverageMeter("TP", ":.4f", Summary.SUM),
                    "FP": AverageMeter("FP", ":.4f", Summary.SUM),
                    "FN": AverageMeter("FN", ":.4f", Summary.SUM)}

        training_model.eval()
        for data_batch in tqdm.tqdm(validation_loader):
            # Prepare data and convert relevant tensors to bfloat16
            data_batch = dict_to_cuda(data_batch)
            for key in ["global_enc_images", "grounding_enc_images"]:
                data_batch[key] = data_batch[key].bfloat16()
            torch.cuda.empty_cache()
            # Model inference without gradient tracking
            with torch.no_grad():
                results = training_model(**data_batch)

            predictions = results["pred_masks"]
            gt_masks = results["gt_masks"][0].int()
            
            # # Skip samples with empty masks (real images)
            # if gt_masks.numel() == 0 or gt_masks.shape[0] == 0:
            #     continue
                
            # Note: An error at this line may suggest that the dataset used for validation does not support
            # segmentation tasks. Ensure that the dataset is appropriate for segmentation analysis.
            predicted_masks = (predictions[0] > 0).int()
            assert len(predictions) == 1

            intersection, union, accuracy_iou = 0.0, 0.0, 0.0
            batch_tp, batch_fp, batch_fn = 0.0, 0.0, 0.0
            
            # # Skip if no masks to evaluate
            # if gt_masks.shape[0] == 0 or predicted_masks.shape[0] == 0:
            #     continue
                
            for target, prediction in zip(gt_masks, predicted_masks):
                intersect, union_, _ = intersectionAndUnionGPU(
                    prediction.contiguous().clone(), target.contiguous(), 2, ignore_index=255
                )
                intersection += intersect
                union += union_
                accuracy_iou += intersect / (union_ + 1e-5)
                # handles no-object targets
                accuracy_iou[union_ == 0] += 1.0
                
                # Compute TP, FP, FN for F1 score (foreground class only)
                pred_fg = (prediction == 1).float()
                target_fg = (target == 1).float()
                batch_tp += (pred_fg * target_fg).sum().item()
                batch_fp += (pred_fg * (1 - target_fg)).sum().item()
                batch_fn += ((1 - pred_fg) * target_fg).sum().item()

            # Only update trackers if we actually computed metrics
            if isinstance(intersection, torch.Tensor):
                intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
                accuracy_iou = accuracy_iou.cpu().numpy() / gt_masks.shape[0]
                trackers["intersection"].update(intersection)
                trackers["union"].update(union)
                trackers["gIoU"].update(accuracy_iou, n=gt_masks.shape[0])
                trackers["TP"].update(batch_tp)
                trackers["FP"].update(batch_fp)
                trackers["FN"].update(batch_fn)

        for meter in trackers.values():
            meter.all_reduce()

        iou_per_class = trackers["intersection"].sum / (trackers["union"].sum + 1e-10)
        class_iou = iou_per_class[1]  # Foreground IoU
        background_iou = iou_per_class[0]  # Background IoU
        mean_iou = (background_iou + class_iou) / 2.0  # mIoU
        global_iou = trackers["gIoU"].avg[1]
        
        # Compute F1 score
        tp_sum = trackers["TP"].sum
        fp_sum = trackers["FP"].sum
        fn_sum = trackers["FN"].sum
        precision = tp_sum / (tp_sum + fp_sum + 1e-10)
        recall = tp_sum / (tp_sum + fn_sum + 1e-10)
        f1_score = 2 * precision * recall / (precision + recall + 1e-10)

        if args.local_rank == 0:
            tensorboard_writer.add_scalar("val/giou", global_iou, current_epoch)
            tensorboard_writer.add_scalar("val/ciou", class_iou, current_epoch)
            tensorboard_writer.add_scalar("val/miou", mean_iou, current_epoch)
            tensorboard_writer.add_scalar("val/f1", f1_score, current_epoch)
            tensorboard_writer.add_scalar("val/precision", precision, current_epoch)
            tensorboard_writer.add_scalar("val/recall", recall, current_epoch)
            print("giou: {:.4f}, ciou: {:.4f}, miou: {:.4f}, F1: {:.4f} (P: {:.4f}, R: {:.4f})".format(
                global_iou, class_iou, mean_iou, f1_score, precision, recall))

        # Restore training mode and clear cache to prevent OOM on next training step
        training_model.train()
        torch.cuda.empty_cache()
        
        # Return unified dict format for consistency
        return {
            "giou": global_iou,
            "ciou": class_iou,
            "miou": mean_iou,
            "f1": f1_score,
            "precision": precision,
            "recall": recall,
        }
    else:
        # Initializing performance trackers
        trackers = {"loss": AverageMeter("Loss", ":.4f"), "ce_loss": AverageMeter("CeLoss", ":.4f"),
                    "mask_bce_loss": AverageMeter("MaskBCELoss", ":.4f"),
                    "mask_dice_loss": AverageMeter("MaskDICELoss", ":.4f"),
                    "mask_loss": AverageMeter("MaskLoss", ":.4f"),
                    "scama_loss": AverageMeter("SCAMALoss", ":.4f")}

        # Prepare model for validation phase
        # Hack to get the loss
        training_model.train()

        for data_batch in tqdm.tqdm(validation_loader):
            # Prepare data and convert relevant tensors to bfloat16
            data_batch = dict_to_cuda(data_batch)
            for key in ["global_enc_images", "grounding_enc_images"]:
                if data_batch[key] is not None:
                    data_batch[key] = data_batch[key].bfloat16()
            torch.cuda.empty_cache()
            # Model inference without gradient tracking
            with torch.no_grad():
                predictions = training_model(**data_batch)
            # Update performance metrics)
            for key, tracker in trackers.items():
                tracker.update(predictions[key].item(), data_batch["global_enc_images"].size(0))

        # Synchronize metrics across processes
        for tracker in trackers.values():
            tracker.all_reduce()
        # Calculate average validation loss
        avg_val_loss = trackers["ce_loss"].avg
        # Tensorboard logging for primary process
        if args.local_rank == 0:
            tensorboard_writer.add_scalar("val/loss", avg_val_loss, current_epoch)
        avg_losses = {key: tracker.avg for key, tracker in trackers.items()}
        
        # Clear cache to prevent OOM on next training step
        torch.cuda.empty_cache()
        
        return avg_losses



    
def notify(start_time, info=""):
    now_str = time.strftime("%H:%M", time.localtime(time.time()))
    print(f"[notify] start={start_time} end={now_str} info={info}")


if __name__ == "__main__":
    start_time = time.time()
    local_time = time.localtime(start_time)
    start_time_str = time.strftime("%H:%M",local_time)
    
    # Setup logging FIRST - before anything else to capture all output
    # Quick parse to get log_base_dir and exp_name for logging
    def quick_parse_for_logging():
        """Quickly extract logging-related args before full parse."""
        log_base_dir = "./output_origin_multi_prompt"  # default
        exp_name = "GlamFinetuneOS"  # default
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        
        for i, arg in enumerate(sys.argv[1:]):
            if arg == "--log_base_dir" and i + 1 < len(sys.argv) - 1:
                log_base_dir = sys.argv[i + 2]
            elif arg == "--exp_name" and i + 1 < len(sys.argv) - 1:
                exp_name = sys.argv[i + 2]
            elif arg == "--local_rank" and i + 1 < len(sys.argv) - 1:
                local_rank = int(sys.argv[i + 2])
        return log_base_dir, exp_name, local_rank
    
    _log_base_dir, _exp_name, _local_rank = quick_parse_for_logging()
    log_dir = os.path.join(_log_base_dir, _exp_name, "logs")
    logger = setup_logging(log_dir, _exp_name, _local_rank)
    
    # Log the full command for reproducibility
    print(f"Command: {' '.join(sys.argv)}")
    
    # Now do full argument parsing (all output will be logged)
    args = parse_args(sys.argv[1:])
    
    # try:
    #     main(args)
    # except Exception as e:
    #     print(e)
    #     notify(start_time_str,e)
        
    main(args)
    notify(start_time_str,"complete")
        
