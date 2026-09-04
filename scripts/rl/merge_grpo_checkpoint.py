#!/usr/bin/env python3
"""
Merge a GRPO LoRA checkpoint into a LaP-Forensic base model for inference.

Usage:
    python scripts/rl/merge_grpo_checkpoint.py \
        --checkpoint_dir ./runs/grpo/<run>/checkpoint-100 \
        --output_dir ./checkpoints/lap_forensic_grpo_merged

Or use the final checkpoint:
    python scripts/rl/merge_grpo_checkpoint.py \
        --checkpoint_dir ./runs/grpo/<run>/final \
        --output_dir ./checkpoints/lap_forensic_grpo_merged
"""

import os
import sys
import json
import argparse
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def parse_args():
    parser = argparse.ArgumentParser(description="Merge GRPO LoRA checkpoint for inference")
    parser.add_argument(
        "--checkpoint_dir", 
        type=str, 
        required=True,
        help="Path to GRPO checkpoint directory (contains adapter_config.json)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True,
        help="Path to save merged model"
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default=None,
        help="Override base model path (default: read from adapter_config.json)"
    )
    parser.add_argument(
        "--vision_pretrained",
        type=str,
        default="./checkpoints/sam_vit_h_4b8939.pth",
        help="Path to SAM vision encoder weights"
    )
    parser.add_argument(
        "--save_full_model",
        action="store_true",
        default=False,
        help="Save complete model including vision tower (larger file size)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("LaP-Forensic GRPO LoRA Checkpoint Merger")
    print("=" * 60)
    
    # Check checkpoint exists
    adapter_config_path = os.path.join(args.checkpoint_dir, "adapter_config.json")
    if not os.path.exists(adapter_config_path):
        print(f"Error: adapter_config.json not found in {args.checkpoint_dir}")
        print("Make sure you're pointing to a valid GRPO checkpoint directory.")
        sys.exit(1)
    
    # Read adapter config to get base model path
    with open(adapter_config_path, "r") as f:
        adapter_config = json.load(f)
    
    base_model_path = args.base_model_path or adapter_config.get("base_model_name_or_path")
    if not base_model_path:
        print("Error: Could not determine base model path")
        sys.exit(1)
    
    print(f"Checkpoint: {args.checkpoint_dir}")
    print(f"Base Model: {base_model_path}")
    print(f"Output Dir: {args.output_dir}")
    print(f"LoRA r={adapter_config.get('r')}, alpha={adapter_config.get('lora_alpha')}")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Import LaP-Forensic model (must be after sys.path setup)
    from model.lap import LaPForCausalLM
    from transformers import AutoTokenizer
    from peft import PeftModel
    
    print("\n[1/5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    
    print("\n[2/5] Loading LaP-Forensic base model...")
    # Load base model with LaP-Forensic-specific config
    model_args = {
        # Required parameters
        'seg_token_idx': tokenizer.convert_tokens_to_ids("[SEG]"),
        'bbox_token_idx': tokenizer.convert_tokens_to_ids("<bbox>"),
        'eop_token_idx': tokenizer.convert_tokens_to_ids("</p>"),
        'bop_token_idx': tokenizer.convert_tokens_to_ids("<p>"),
        # Vision
        'vision_tower': 'openai/clip-vit-large-patch14-336',
        'vision_pretrained': args.vision_pretrained,
        'use_mm_start_end': True,
        'mm_use_im_start_end': True,
        'mm_vision_select_layer': -2,
        'with_region': True,
        # Dual-stream consistency
        'use_consistency_stream': True,
        'vae_path': 'stabilityai/sd-vae-ft-mse',
        'consistency_encoder_type': 'clip',
        # Loss weights (not used for inference but required)
        'ce_loss_weight': 1.0,
        'dice_loss_weight': 0.0,
        'bce_loss_weight': 0.0,
        'scama_loss_weight': 0.0,
        # Other
        'train_mask_decoder': False,
        'out_dim': 256,
    }
    
    print(f"  seg_token_idx: {model_args['seg_token_idx']}")
    print(f"  bbox_token_idx: {model_args['bbox_token_idx']}")
    
    base_model = LaPForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        **model_args
    )
    
    print("\n[3/5] Initializing vision modules...")
    base_model.get_model().initialize_vision_modules(base_model.get_model().config)
    vision_tower = base_model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch.bfloat16)
    
    print("\n[4/5] Loading and merging LoRA adapter...")
    # Load PEFT adapter
    model = PeftModel.from_pretrained(
        base_model,
        args.checkpoint_dir,
        torch_dtype=torch.bfloat16,
    )
    
    # Merge LoRA weights into base model
    print("Merging LoRA weights...")
    model = model.merge_and_unload()
    print("LoRA merge complete!")
    
    print("\n[5/5] Saving merged model...")
    
    # Save model
    if args.save_full_model:
        print("Saving complete model (including vision tower)...")
        model.save_pretrained(args.output_dir)
    else:
        # Filter out vision tower to reduce file size
        print("Saving model (excluding vision tower for smaller size)...")
        state_dict = {}
        for k, v in model.state_dict().items():
            if "vision_tower" not in k:
                state_dict[k] = v
        model.save_pretrained(args.output_dir, state_dict=state_dict)
    
    # Save tokenizer
    tokenizer.save_pretrained(args.output_dir)
    
    # Copy config.json from base model (important for loading)
    import shutil
    base_config_path = os.path.join(base_model_path, "config.json")
    if os.path.exists(base_config_path):
        shutil.copy(base_config_path, os.path.join(args.output_dir, "config.json"))
        print("Copied config.json from base model")
    
    # Save merge info
    merge_info = {
        "checkpoint_dir": args.checkpoint_dir,
        "base_model_path": base_model_path,
        "lora_r": adapter_config.get("r"),
        "lora_alpha": adapter_config.get("lora_alpha"),
        "target_modules": adapter_config.get("target_modules"),
        "merged_with_script": "scripts/rl/merge_grpo_checkpoint.py",
        "model_type": "llava_lap",
        "architecture": "LaPForCausalLM",
    }
    with open(os.path.join(args.output_dir, "merge_info.json"), "w") as f:
        json.dump(merge_info, f, indent=2)
    
    print("\n" + "=" * 60)
    print(f"✅ Merged model saved to: {args.output_dir}")
    print("=" * 60)
    print("\nTo use for inference:")
    print("```python")
    print("from model.lap import LaPForCausalLM")
    print(f"model = LaPForCausalLM.from_pretrained('{args.output_dir}')")
    print("```")


if __name__ == "__main__":
    main()
