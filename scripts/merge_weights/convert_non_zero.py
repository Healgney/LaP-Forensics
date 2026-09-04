import torch
import argparse
import os
import sys

def convert(base_dir, tag, output_file):
    checkpoint_dir = os.path.join(base_dir, tag)
    model_file = os.path.join(checkpoint_dir, "mp_rank_00_model_states.pt")
    
    if not os.path.exists(model_file):
        print(f"Error: Cannot find {model_file}")
        # Fallback to check if it is directly in base_dir if tag is empty or misconfigured
        if os.path.exists(os.path.join(base_dir, "mp_rank_00_model_states.pt")):
             model_file = os.path.join(base_dir, "mp_rank_00_model_states.pt")
             print(f"Found model file at {model_file}")
        else:
             sys.exit(1)
    
    print(f"Loading {model_file}...")
    try:
        checkpoint = torch.load(model_file, map_location="cpu")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        sys.exit(1)
    
    if "module" not in checkpoint:
        print("Error: Checkpoint does not contain 'module' key")
        sys.exit(1)
        
    state_dict = checkpoint["module"]
    
    # Convert to fp32
    print("Converting to fp32...")
    new_state_dict = {}
    for k, v in state_dict.items():
        # Remove 'module.' prefix if present (DataParallel/DDP sometimes adds it, but here it seems the key 'module' contains the dict)
        # Wait, from my inspection: 'base_model.model.model.embed_tokens.weight'
        # It seems it doesn't have an extra 'module.' prefix inside the dict, but the dict itself is under 'module' key of the checkpoint.
        new_state_dict[k] = v.float()
        
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Saving to {output_file}...")
    torch.save(new_state_dict, output_file)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("base_dir", type=str, help="Base directory containing the checkpoint folder")
    parser.add_argument("output_file", type=str, help="Path to save the pytorch_model.bin")
    parser.add_argument("--tag", type=str, default="", help="Checkpoint tag (e.g., global_step1152)")
    args = parser.parse_args()
    
    convert(args.base_dir, args.tag, args.output_file)
