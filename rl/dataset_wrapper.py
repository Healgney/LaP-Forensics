"""
TRL Dataset Wrapper for LaP-Forensic

Wraps LaPGCGDataset to provide TRL-compatible format for GRPOTrainer.

Design: The LaPReward class holds a reference to this dataset and
uses idx from TRL kwargs to look up gt_masks and images for reward computation.
"""

import os
import time
import logging
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class TRLDatasetWrapper(Dataset):
    """
    Wraps LaPGCGDataset for TRL GRPOTrainer compatibility.
    
    TRL expects datasets to return dict with 'prompt' key.
    This wrapper extracts prompts and stores metadata for reward computation.
    """
    
    def __init__(
        self,
        base_dataset,
        tokenizer,
        conversation_template: str = "llava_v1",
    ):
        """
        Args:
            base_dataset: LaPGCGDataset or similar
            tokenizer: Tokenizer for formatting
            conversation_template: Conversation template name
        """
        self.base_dataset = base_dataset
        self.tokenizer = tokenizer
        self.conversation_template = conversation_template
        
        # Store metadata for reward computation
        self._metadata_cache = {}
        self.debug = os.getenv("LAP_FORENSIC_DEBUG", "0") != "0"
        try:
            self.debug_every = int(os.getenv("LAP_FORENSIC_DEBUG_EVERY", "10"))
        except ValueError:
            self.debug_every = 10
        self._call_count = 0
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get item in TRL-compatible format.
        
        Returns dict with:
            - prompt: str (the input prompt)
            - prompt_ids: tensor (tokenized prompt)
            
        Metadata stored separately for reward computation.
        """
        self._call_count += 1
        start = time.perf_counter()
        # Get raw data from base dataset
        raw_data = self.base_dataset[idx]
        
        # Unpack (format depends on base dataset)
        # LaPGCGDataset returns tuple:
        # (image_path, global_enc_image, grounding_enc_image, bboxes, 
        #  conversations, masks, label, resize, questions, sampled_classes)
        (image_path, global_enc_image, grounding_enc_image, bboxes,
         conversations, masks, label, resize, questions, sampled_classes) = raw_data
        
        # Extract prompt (instruction part only)
        # conversations is a list of full conversation strings
        if conversations:
            full_conv = conversations[0]
            # Extract just the prompt part (before assistant response)
            # Handle different conversation formats
            if "###Assistant:" in full_conv:
                # Format: ###Human: ... ###Assistant: ...
                prompt = full_conv.split("###Assistant:")[0] + "###Assistant:"
            elif "ASSISTANT:" in full_conv:
                # Format: USER: ... ASSISTANT: ...
                prompt = full_conv.split("ASSISTANT:")[0] + "ASSISTANT:"
            else:
                prompt = full_conv
            
            # CRITICAL: Wrap <image> token with <im_start> and <im_end>
            # This is required for the model to properly process images
            from tools.utils import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
            if DEFAULT_IMAGE_TOKEN in prompt and DEFAULT_IM_START_TOKEN not in prompt:
                replace_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
                prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)
        else:
            prompt = ""
        
        # Store metadata for reward computation
        # LaPReward class will access this via _metadata_cache using idx
        self._metadata_cache[idx] = {
            'image_path': image_path,
            'global_enc_image': global_enc_image,
            'grounding_enc_image': grounding_enc_image,
            'bboxes': bboxes,
            'masks': masks,  # GT masks for reward computation
            'label': label,
            'resize': resize,
            'full_conversation': conversations[0] if conversations else "",
        }
        
        # For LaPGRPOTrainer, we pass pre-encoded CLIP images
        # This allows the model to generate with proper visual context
        item = {
            'prompt': prompt,
            'idx': idx,  # Used by LaPReward to look up metadata
            'global_enc_image': global_enc_image,  # Pre-encoded CLIP image
            'grounding_enc_image': grounding_enc_image,  # Pre-encoded SAM image
        }
        
        if self.debug and self._call_count % self.debug_every == 1:
            elapsed = time.perf_counter() - start
            logger.info("Dataset debug: __getitem__ idx=%d time=%.3fs prompt_len=%d", idx, elapsed, len(prompt))
        
        return item
    
    def get_metadata(self, idx: int) -> Dict[str, Any]:
        """Retrieve stored metadata for reward computation."""
        return self._metadata_cache.get(idx, {})
    
    def collate_fn(self, batch: List[Dict]) -> Dict[str, Any]:
        """
        Collate function for DataLoader.
        
        Args:
            batch: List of items from __getitem__
            
        Returns:
            Batched dict with prompts and metadata
        """
        start = time.perf_counter()
        prompts = [item['prompt'] for item in batch]
        indices = [item['idx'] for item in batch]
        
        # Collect metadata
        metadata = {
            'indices': indices,
            'global_enc_images': [],
            'grounding_enc_images': [],
            'masks_list': [],
            'label_list': [],
            'resize_list': [],
        }
        
        for idx in indices:
            meta = self.get_metadata(idx)
            if meta:
                metadata['global_enc_images'].append(meta.get('global_enc_image'))
                metadata['grounding_enc_images'].append(meta.get('grounding_enc_image'))
                metadata['masks_list'].append(meta.get('masks'))
                metadata['label_list'].append(meta.get('label'))
                metadata['resize_list'].append(meta.get('resize'))
        
        # Stack tensors where possible
        if metadata['global_enc_images'] and metadata['global_enc_images'][0] is not None:
            metadata['global_enc_images'] = torch.stack(metadata['global_enc_images'])
        if metadata['grounding_enc_images'] and metadata['grounding_enc_images'][0] is not None:
            metadata['grounding_enc_images'] = torch.stack(metadata['grounding_enc_images'])
        
        output = {
            'prompts': prompts,
            **metadata,
        }
        
        if self.debug and self._call_count % self.debug_every == 1:
            elapsed = time.perf_counter() - start
            logger.info("Dataset debug: collate_fn batch=%d time=%.3fs", len(batch), elapsed)
        
        return output


def create_trl_dataset(
    dataset_dir: str,
    tokenizer,
    global_image_encoder: str,
    epoch_samples: int = 8000,
    precision: str = "bf16",
    image_size: int = 512,
    num_classes_per_sample: int = 3,
    validation: bool = False,
    use_dual_stream: bool = True,  # MANDATORY: SFT was trained with dual-stream
):
    """
    Create a TRL-compatible dataset from LaPGCGDataset.
    
    Args:
        dataset_dir: Path to dataset directory
        tokenizer: Tokenizer
        global_image_encoder: Vision encoder name
        epoch_samples: Samples per epoch
        precision: Training precision
        image_size: Image size
        num_classes_per_sample: Classes per sample
        validation: Whether this is validation set
        use_dual_stream: Enable dual-stream (MUST match SFT training)
    
    Returns:
        TRLDatasetWrapper instance
    """
    from dataset.gcg_datasets.GranDf_gcg_ds import LaPGCGDataset
    
    base_dataset = LaPGCGDataset(
        dataset_dir=dataset_dir,
        tokenizer=tokenizer,
        global_image_encoder=global_image_encoder,
        epoch_samples=epoch_samples,
        precision=precision,
        image_size=image_size,
        num_classes_per_sample=num_classes_per_sample,
        validation=validation,
        use_dual_stream=use_dual_stream,  # Pass dual-stream flag
    )
    
    return TRLDatasetWrapper(base_dataset, tokenizer)

