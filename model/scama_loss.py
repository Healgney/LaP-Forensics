"""
Self-Consistent Attention-Mask Alignment (SCAMA) Loss

This module implements SCAMA loss to enforce spatial consistency between
cross-attention maps and ground truth segmentation masks, addressing hallucination
issues in MLLM explanations for deepfake detection.

Reference: SCAMA Loss enforces that explanation tokens' attention aligns with
the predicted forgery localization masks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


def compute_attention_iou(attention_map: torch.Tensor, gt_mask: torch.Tensor, 
                          threshold: float = 0.5) -> torch.Tensor:
    """
    Compute IoU between attention heatmap and ground truth mask.
    
    Args:
        attention_map: Attention weights [H, W] or [B, H, W], normalized to [0, 1]
        gt_mask: Ground truth binary mask [H, W] or [B, H, W], values in {0, 1}
        threshold: Threshold to binarize attention map
        
    Returns:
        IoU score(s) as tensor
    """
    # Ensure same spatial dimensions
    if attention_map.shape != gt_mask.shape:
        # Resize attention map to match GT mask size
        if attention_map.dim() == 2:
            attention_map = attention_map.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            needs_squeeze = True
        else:
            attention_map = attention_map.unsqueeze(1)  # [B, 1, H, W]
            needs_squeeze = False
            
        attention_map = F.interpolate(
            attention_map, 
            size=gt_mask.shape[-2:], 
            mode='bilinear', 
            align_corners=False
        )
        
        if needs_squeeze:
            attention_map = attention_map.squeeze(0).squeeze(0)
        else:
            attention_map = attention_map.squeeze(1)
    
    # Binarize attention map
    attention_binary = (attention_map > threshold).float()
    
    # Compute intersection and union
    intersection = (attention_binary * gt_mask).sum(dim=(-2, -1))
    union = (attention_binary + gt_mask).clamp(0, 1).sum(dim=(-2, -1))
    
    # Compute IoU with epsilon for numerical stability
    iou = (intersection + 1e-6) / (union + 1e-6)
    
    return iou


def extract_explanation_tokens(input_ids: torch.Tensor, 
                               seg_token_idx: int,
                               special_token_indices: Optional[List[int]] = None) -> torch.Tensor:
    """
    Identify explanation tokens by excluding SEG tokens, image tokens, and prompt tokens.
    
    Args:
        input_ids: Token IDs [B, seq_len]
        seg_token_idx: Index of [SEG] token
        special_token_indices: List of special token indices to exclude (e.g., <image>, <bbox>)
        
    Returns:
        Boolean mask [B, seq_len] where True indicates explanation tokens
    """
    batch_size, seq_len = input_ids.shape
    
    # Start with all tokens as potential explanation tokens
    explanation_mask = torch.ones_like(input_ids, dtype=torch.bool)
    
    # Exclude SEG tokens
    explanation_mask = explanation_mask & (input_ids != seg_token_idx)
    
    # Exclude special tokens if provided
    if special_token_indices is not None:
        for special_idx in special_token_indices:
            explanation_mask = explanation_mask & (input_ids != special_idx)
    
    # Exclude padding tokens (assuming 0 is pad token)
    explanation_mask = explanation_mask & (input_ids != 0)
    
    return explanation_mask


def aggregate_attention_maps(attentions: Tuple[torch.Tensor, ...], 
                             token_positions: torch.Tensor,
                             image_token_length: int = 576) -> torch.Tensor:
    """
    Aggregate cross-attention maps across layers and heads for specific tokens.
    
    Args:
        attentions: Tuple of attention tensors from transformer layers
                   Each tensor has shape [B, num_heads, seq_len, seq_len]
        token_positions: Boolean mask [B, seq_len] indicating which tokens to aggregate
        image_token_length: Number of image tokens (default 576 for 24x24 patches)
        
    Returns:
        Aggregated attention map [B, num_tokens, sqrt(image_len), sqrt(image_len)]
    """
    # Use last layer's attention (can be extended to multi-layer aggregation)
    last_layer_attn = attentions[-1]  # [B, num_heads, seq_len, seq_len]
    
    batch_size, num_heads, seq_len, _ = last_layer_attn.shape
    
    # Average across attention heads
    avg_attn = last_layer_attn.mean(dim=1)  # [B, seq_len, seq_len]
    
    # Extract attention to image tokens (assuming image tokens are at the beginning)
    # Handle case where seq_len < image_token_length
    actual_image_len = min(seq_len, image_token_length)
    # Shape: [B, seq_len, actual_image_len]
    attn_to_image = avg_attn[:, :, :actual_image_len]
    
    # Select only explanation token rows
    # Get indices of explanation tokens for each batch
    explanation_attns = []
    for b in range(batch_size):
        token_indices = torch.where(token_positions[b])[0]
        if len(token_indices) > 0:
            # [num_explanation_tokens, actual_image_len]
            batch_attn = attn_to_image[b, token_indices, :]
            
            # If actual_image_len < image_token_length, pad with zeros
            if actual_image_len < image_token_length:
                padding = torch.zeros(
                    batch_attn.shape[0], 
                    image_token_length - actual_image_len,
                    device=batch_attn.device
                )
                batch_attn = torch.cat([batch_attn, padding], dim=1)
            
            explanation_attns.append(batch_attn)
        else:
            # No explanation tokens, create dummy
            explanation_attns.append(torch.zeros(1, image_token_length, device=attn_to_image.device))
    
    # Reshape to spatial dimensions (assuming square image tokens)
    spatial_size = int(image_token_length ** 0.5)
    assert spatial_size * spatial_size == image_token_length, \
        f"Image token length {image_token_length} is not a perfect square"
    
    # Stack and reshape
    aggregated_maps = []
    for attn in explanation_attns:
        # [num_tokens, image_token_length] -> [num_tokens, H, W]
        spatial_attn = attn.view(-1, spatial_size, spatial_size)
        aggregated_maps.append(spatial_attn)
    
    return aggregated_maps


def scama_loss(attentions: Tuple[torch.Tensor, ...],
               input_ids: torch.Tensor,
               gt_masks: List[torch.Tensor],
               seg_token_idx: int,
               special_token_indices: Optional[List[int]] = None,
               image_token_length: int = 576,
               attention_threshold: float = 0.5) -> torch.Tensor:
    """
    Compute SCAMA loss: L_SCAMA = Σ_{t ∈ T_exp} (1 - IoU(AttnMap(t), GT_Mask))
    
    Args:
        attentions: Tuple of attention tensors from transformer layers
        input_ids: Token IDs [B, seq_len]
        gt_masks: List of ground truth masks, one per batch item [H, W]
        seg_token_idx: Index of [SEG] token
        special_token_indices: List of special token indices to exclude
        image_token_length: Number of image tokens
        attention_threshold: Threshold for binarizing attention maps
        
    Returns:
        SCAMA loss value (scalar tensor)
    """
    if attentions is None or len(attentions) == 0:
        # No attention maps available, return zero loss
        return torch.tensor(0.0, device=input_ids.device)
    
    batch_size = input_ids.shape[0]
    
    # Extract explanation tokens
    explanation_mask = extract_explanation_tokens(
        input_ids, seg_token_idx, special_token_indices
    )
    
    # Aggregate attention maps for explanation tokens
    aggregated_attns = aggregate_attention_maps(
        attentions, explanation_mask, image_token_length
    )
    
    # Compute IoU loss for each batch item
    total_loss = 0.0
    total_tokens = 0
    
    for b in range(batch_size):
        if b >= len(gt_masks) or gt_masks[b] is None:
            continue
            
        gt_mask = gt_masks[b]  # [H, W] or [num_masks, H, W]
        
        # If multiple GT masks, take the union or first one
        if gt_mask.dim() == 3:
            gt_mask = gt_mask.max(dim=0)[0]  # Union of all masks
        
        attn_maps = aggregated_attns[b]  # [num_tokens, H_attn, W_attn]
        num_tokens = attn_maps.shape[0]
        
        if num_tokens == 0:
            continue
        
        # Compute IoU for each explanation token's attention map
        for token_idx in range(num_tokens):
            attn_map = attn_maps[token_idx]  # [H_attn, W_attn]
            
            # Compute IoU
            iou = compute_attention_iou(attn_map, gt_mask, attention_threshold)
            
            # Loss is 1 - IoU (we want to maximize IoU, minimize loss)
            total_loss += (1.0 - iou)
            total_tokens += 1
    
    # Average loss across all explanation tokens
    if total_tokens > 0:
        scama_loss_value = total_loss / total_tokens
    else:
        scama_loss_value = torch.tensor(0.0, device=input_ids.device)
    
    return scama_loss_value


class SCAMALoss(nn.Module):
    """
    SCAMA Loss module for enforcing attention-mask alignment.
    """
    
    def __init__(self, 
                 seg_token_idx: int,
                 special_token_indices: Optional[List[int]] = None,
                 image_token_length: int = 576,
                 attention_threshold: float = 0.5):
        """
        Args:
            seg_token_idx: Index of [SEG] token
            special_token_indices: List of special token indices to exclude
            image_token_length: Number of image tokens
            attention_threshold: Threshold for binarizing attention maps
        """
        super().__init__()
        self.seg_token_idx = seg_token_idx
        self.special_token_indices = special_token_indices
        self.image_token_length = image_token_length
        self.attention_threshold = attention_threshold
    
    def forward(self,
                attentions: Tuple[torch.Tensor, ...],
                input_ids: torch.Tensor,
                gt_masks: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute SCAMA loss.
        
        Args:
            attentions: Tuple of attention tensors from transformer layers
            input_ids: Token IDs [B, seq_len]
            gt_masks: List of ground truth masks
            
        Returns:
            SCAMA loss value
        """
        return scama_loss(
            attentions=attentions,
            input_ids=input_ids,
            gt_masks=gt_masks,
            seg_token_idx=self.seg_token_idx,
            special_token_indices=self.special_token_indices,
            image_token_length=self.image_token_length,
            attention_threshold=self.attention_threshold
        )
