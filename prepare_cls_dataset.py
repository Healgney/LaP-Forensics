#!/usr/bin/env python3
"""
Prepare Classification Dataset for LaP-Forensic Deepfake Detection

This script helps you:
1. Generate annotation JSON files from image directories
2. Combine fake images (from SynthScars) with real images (from COCO/ImageNet/LSUN)
3. Split into train/test sets

Usage:
    python prepare_cls_dataset.py \
        --fake_dir /path/to/fake/images \
        --real_dir /path/to/real/images \
        --output_dir /path/to/output \
        --train_ratio 0.8

Expected output structure:
    output_dir/
    ├── train.json
    ├── test.json
    ├── fake/  (symlinks or copies)
    └── real/  (symlinks or copies)
"""

import os
import json
import argparse
import random
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm


def get_image_files(directory: str, extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')) -> List[str]:
    """Get all image files from a directory recursively."""
    image_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(extensions):
                rel_path = os.path.relpath(os.path.join(root, file), directory)
                image_files.append(rel_path)
    return image_files


def create_annotation_list(
    fake_images: List[str],
    real_images: List[str],
    fake_prefix: str = "fake/",
    real_prefix: str = "real/",
    is_test: bool = False
) -> List[Dict]:
    """
    Create annotation list in the format required by LaPClsDataset.
    
    Note: The train.py uses different field names for train vs test:
    - Train set uses 'image_path'
    - Test set uses 'image'
    
    Args:
        fake_images: List of fake image paths
        real_images: List of real image paths
        fake_prefix: Prefix to add to fake image paths
        real_prefix: Prefix to add to real image paths
        is_test: If True, use 'image' field instead of 'image_path'
    """
    annotations = []
    # Use 'image' for test set, 'image_path' for train set (per train.py line 106)
    path_key = "image" if is_test else "image_path"
    
    # Add fake images
    for img_path in fake_images:
        annotations.append({
            path_key: os.path.join(fake_prefix, img_path),
            "label": "fake"
        })
    
    # Add real images
    for img_path in real_images:
        annotations.append({
            path_key: os.path.join(real_prefix, img_path),
            "label": "real"
        })
    
    return annotations


def balance_dataset(fake_images: List[str], real_images: List[str], balance_strategy: str = "undersample") -> Tuple[List[str], List[str]]:
    """Balance the dataset between fake and real images."""
    n_fake = len(fake_images)
    n_real = len(real_images)
    
    if balance_strategy == "undersample":
        # Undersample the majority class
        min_count = min(n_fake, n_real)
        fake_images = random.sample(fake_images, min_count)
        real_images = random.sample(real_images, min_count)
    elif balance_strategy == "oversample":
        # Oversample the minority class
        max_count = max(n_fake, n_real)
        if n_fake < max_count:
            fake_images = fake_images + random.choices(fake_images, k=max_count - n_fake)
        if n_real < max_count:
            real_images = real_images + random.choices(real_images, k=max_count - n_real)
    # else: no balancing
    
    return fake_images, real_images


def split_train_test(items: List, train_ratio: float = 0.8) -> Tuple[List, List]:
    """Split items into train and test sets."""
    random.shuffle(items)
    split_idx = int(len(items) * train_ratio)
    return items[:split_idx], items[split_idx:]


def main():
    parser = argparse.ArgumentParser(description="Prepare classification dataset for LaP-Forensic")
    parser.add_argument("--fake_dir", type=str, required=True, help="Directory containing fake images")
    parser.add_argument("--real_dir", type=str, required=True, help="Directory containing real images")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for annotations")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Train/test split ratio")
    parser.add_argument("--balance", type=str, default="undersample", choices=["undersample", "oversample", "none"],
                        help="Dataset balancing strategy")
    parser.add_argument("--max_images", type=int, default=None, help="Maximum number of images per class")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fake_prefix", type=str, default="fake/", help="Prefix for fake image paths in JSON")
    parser.add_argument("--real_prefix", type=str, default="real/", help="Prefix for real image paths in JSON")
    
    args = parser.parse_args()
    random.seed(args.seed)
    
    # Get image files
    print(f"Scanning for fake images in: {args.fake_dir}")
    fake_images = get_image_files(args.fake_dir)
    print(f"Found {len(fake_images)} fake images")
    
    print(f"Scanning for real images in: {args.real_dir}")
    real_images = get_image_files(args.real_dir)
    print(f"Found {len(real_images)} real images")
    
    # Limit images if specified
    if args.max_images:
        if len(fake_images) > args.max_images:
            fake_images = random.sample(fake_images, args.max_images)
        if len(real_images) > args.max_images:
            real_images = random.sample(real_images, args.max_images)
        print(f"Limited to {len(fake_images)} fake and {len(real_images)} real images")
    
    # Balance dataset
    if args.balance != "none":
        fake_images, real_images = balance_dataset(fake_images, real_images, args.balance)
        print(f"After {args.balance}: {len(fake_images)} fake and {len(real_images)} real images")
    
    # Split into train/test
    fake_train, fake_test = split_train_test(fake_images, args.train_ratio)
    real_train, real_test = split_train_test(real_images, args.train_ratio)
    
    print(f"Train set: {len(fake_train)} fake + {len(real_train)} real = {len(fake_train) + len(real_train)} total")
    print(f"Test set: {len(fake_test)} fake + {len(real_test)} real = {len(fake_test) + len(real_test)} total")
    
    # Create annotations
    # Note: test JSON uses 'image' field, train JSON uses 'image_path' (per train.py line 106)
    train_annotations = create_annotation_list(fake_train, real_train, args.fake_prefix, args.real_prefix, is_test=False)
    test_annotations = create_annotation_list(fake_test, real_test, args.fake_prefix, args.real_prefix, is_test=True)
    
    # Shuffle annotations
    random.shuffle(train_annotations)
    random.shuffle(test_annotations)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save annotations
    train_json_path = os.path.join(args.output_dir, "train.json")
    test_json_path = os.path.join(args.output_dir, "test.json")
    
    with open(train_json_path, 'w') as f:
        json.dump(train_annotations, f, indent=2)
    print(f"Saved train annotations to: {train_json_path}")
    
    with open(test_json_path, 'w') as f:
        json.dump(test_annotations, f, indent=2)
    print(f"Saved test annotations to: {test_json_path}")
    
    # Print sample entries
    print("\n--- Sample train entries ---")
    for entry in train_annotations[:3]:
        print(f"  {entry}")
    
    print("\n--- Sample test entries ---")
    for entry in test_annotations[:3]:
        print(f"  {entry}")
    
    print("\n✅ Dataset preparation complete!")
    print(f"\nTo use these annotations, run training with:")
    print(f"  --train_json_file {train_json_path}")
    print(f"  --test_json_file {test_json_path}")
    print(f"  --data_base_train /path/to/your/images  # Parent folder containing fake/ and real/ subdirs")
    print(f"  --data_base_test /path/to/your/images")


def create_from_synthscars(synthscars_dir: str, real_images_dir: str, output_dir: str):
    """
    Convenience function to create dataset from SynthScars + real images.
    
    This function is designed specifically for the LaP-Forensic project structure:
    - SynthScars images are treated as 'fake'
    - User provides a directory of real images
    
    Example usage:
        from prepare_cls_dataset import create_from_synthscars
        create_from_synthscars(
            synthscars_dir="/path/to/SynthScars/train/images",
            real_images_dir="/path/to/coco/train2017",
            output_dir="/path/to/output"
        )
    """
    # Get SynthScars images (all fake)
    fake_images = get_image_files(synthscars_dir)
    real_images = get_image_files(real_images_dir)
    
    # Balance
    min_count = min(len(fake_images), len(real_images))
    fake_images = random.sample(fake_images, min_count)
    real_images = random.sample(real_images, min_count)
    
    # Split
    fake_train, fake_test = split_train_test(fake_images, 0.8)
    real_train, real_test = split_train_test(real_images, 0.8)
    
    # Create annotations with absolute paths relative to a common base
    train_annotations = []
    test_annotations = []
    
    for img in fake_train:
        train_annotations.append({
            "image_path": os.path.join(synthscars_dir, img),
            "label": "fake"
        })
    for img in real_train:
        train_annotations.append({
            "image_path": os.path.join(real_images_dir, img),
            "label": "real"
        })
    
    # Note: test set uses 'image' field, not 'image_path'
    for img in fake_test:
        test_annotations.append({
            "image": os.path.join(synthscars_dir, img),
            "label": "fake"
        })
    for img in real_test:
        test_annotations.append({
            "image": os.path.join(real_images_dir, img),
            "label": "real"
        })
    
    random.shuffle(train_annotations)
    random.shuffle(test_annotations)
    
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "train.json"), 'w') as f:
        json.dump(train_annotations, f, indent=2)
    
    with open(os.path.join(output_dir, "test.json"), 'w') as f:
        json.dump(test_annotations, f, indent=2)
    
    print(f"Created dataset with {len(train_annotations)} train and {len(test_annotations)} test samples")


if __name__ == "__main__":
    main()
