"""
Reward Functions for TRL GRPO Training

This module provides TRL-compatible reward functions for GRPO training
of the LaP-Forensic forensic detection model.

Key Design: Class-based LaPReward that holds references to:
- dataset: To look up gt_masks and pixel_values using idx
- model: To run forward pass and extract [SEG] hidden state
- tokenizer: For text processing

Usage with TRL GRPOTrainer:
    reward_module = LaPReward(dataset, model, tokenizer, ...)
    trainer = GRPOTrainer(
        ...,
        reward_funcs=reward_module,  # Uses __call__
    )
"""

import os
import re
import time
import logging
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)


# ============================================================================
# Utility Functions
# ============================================================================

def compute_iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor, 
                threshold: float = 0.5) -> float:
    """
    Compute IoU between predicted and ground truth masks.
    
    Args:
        pred_mask: Predicted mask [H, W]
        gt_mask: Ground truth mask [H, W]
        threshold: Binarization threshold
    
    Returns:
        IoU score as float
    """
    # Handle None or empty masks
    if pred_mask is None or gt_mask is None:
        return 0.0
    
    # Convert list of masks to single tensor if needed
    if isinstance(gt_mask, (list, tuple)):
        if len(gt_mask) == 0:
            return 0.0
        # Take first mask or combine them
        gt_mask = gt_mask[0] if len(gt_mask) == 1 else torch.stack([torch.tensor(m) if not isinstance(m, torch.Tensor) else m for m in gt_mask]).max(dim=0)[0]
    
    if isinstance(pred_mask, (list, tuple)):
        if len(pred_mask) == 0:
            return 0.0
        pred_mask = pred_mask[0] if len(pred_mask) == 1 else torch.stack(pred_mask).max(dim=0)[0]
    
    # Convert numpy to tensor if needed
    if not isinstance(gt_mask, torch.Tensor):
        gt_mask = torch.tensor(gt_mask)
    if not isinstance(pred_mask, torch.Tensor):
        pred_mask = torch.tensor(pred_mask)
    
    # Squeeze extra dimensions (with safety limit to avoid infinite loop)
    for _ in range(10):
        if pred_mask.dim() <= 2:
            break
        pred_mask = pred_mask.squeeze(0)
    
    for _ in range(10):
        if gt_mask.dim() <= 2:
            break
        gt_mask = gt_mask.squeeze(0)
    
    # If still > 2D, take first slice
    if pred_mask.dim() > 2:
        pred_mask = pred_mask[0]
    if gt_mask.dim() > 2:
        gt_mask = gt_mask[0]
    
    # Ensure on same device
    if pred_mask.device != gt_mask.device:
        gt_mask = gt_mask.to(pred_mask.device)
    
    # Apply sigmoid if logits
    if pred_mask.min() < 0 or pred_mask.max() > 1:
        pred_mask = torch.sigmoid(pred_mask)
    
    # Binarize
    pred_binary = (pred_mask > threshold).float()
    gt_binary = gt_mask.float()
    
    # Resize if needed
    if pred_binary.shape != gt_binary.shape:
        pred_binary = F.interpolate(
            pred_binary.unsqueeze(0).unsqueeze(0),
            size=gt_binary.shape[-2:],
            mode='bilinear',
            align_corners=False
        ).squeeze()
    
    # Compute IoU
    intersection = (pred_binary * gt_binary).sum()
    union = (pred_binary + gt_binary).clamp(0, 1).sum()
    
    if union == 0:
        return 1.0  # Both empty
    
    return (intersection / (union + 1e-6)).item()


def compute_iou_matrix(
    pred_masks: List[torch.Tensor],
    gt_masks: List[torch.Tensor],
    threshold: float = 0.5,
) -> torch.Tensor:
    """
    Compute M x N IoU matrix between M predicted masks and N GT masks.
    
    Args:
        pred_masks: List of M predicted mask tensors [H, W]
        gt_masks: List of N ground truth mask tensors [H, W]
        threshold: Binarization threshold for predictions
    
    Returns:
        IoU matrix of shape [M, N]
    """
    M = len(pred_masks)
    N = len(gt_masks)
    iou_matrix = torch.zeros(M, N)
    
    for i, pred in enumerate(pred_masks):
        for j, gt in enumerate(gt_masks):
            iou_matrix[i, j] = compute_iou(pred, gt, threshold=threshold)
    
    return iou_matrix


def greedy_match(iou_matrix: torch.Tensor) -> List[Tuple[int, int]]:
    """
    Greedy best-first bipartite matching on an IoU matrix.
    
    Repeatedly picks the highest IoU pair and removes both from consideration.
    O(M*N*min(M,N)), sufficient for small M, N (typically <10).
    
    Args:
        iou_matrix: [M, N] IoU matrix
    
    Returns:
        List of (pred_idx, gt_idx) matched pairs
    """
    if iou_matrix.numel() == 0:
        return []
    
    M, N = iou_matrix.shape
    mat = iou_matrix.clone()
    matches = []
    
    num_matches = min(M, N)
    for _ in range(num_matches):
        # Find the best remaining pair
        max_val = mat.max()
        if max_val <= 0:
            break  # No positive IoU pairs left
        flat_idx = mat.argmax().item()
        pi, gj = flat_idx // N, flat_idx % N
        matches.append((pi, gj))
        # Remove this pred and gt from future consideration
        mat[pi, :] = -1.0
        mat[:, gj] = -1.0
    
    return matches


def compute_multi_mask_reward(
    pred_masks: List[Optional[torch.Tensor]],
    gt_masks: List[torch.Tensor],
    iou_bonus_threshold: float = 0.7,
    iou_bonus_value: float = 0.5,
    fp_penalty: float = -0.5,
    fn_penalty: float = -0.5,
) -> float:
    """
    Compute mask reward using 1-to-1 bipartite matching between
    predicted masks and ground truth masks.
    
    Args:
        pred_masks: List of M predicted masks (may contain None for failed preds)
        gt_masks: List of N ground truth masks
        iou_bonus_threshold: IoU threshold for bonus
        iou_bonus_value: Bonus value when IoU exceeds threshold
        fp_penalty: Penalty per unmatched prediction (false positive)
        fn_penalty: Penalty per unmatched ground truth (false negative)
    
    Returns:
        Multi-mask reward score (not clipped)
    """
    # Filter out None predictions
    valid_preds = [m for m in pred_masks if m is not None]
    M = len(valid_preds)
    N = len(gt_masks)
    
    # Edge cases
    if M == 0 and N == 0:
        return 1.0  # Both empty — correct
    if M == 0:
        # No predictions but GT exists → all false negatives
        return fn_penalty * N / max(1, N)  # = fn_penalty
    if N == 0:
        # Predictions but no GT → all false positives
        return fp_penalty * M  # Harsh — should not predict anything
    
    # Build IoU matrix [M, N]
    iou_mat = compute_iou_matrix(valid_preds, gt_masks)
    
    # Greedy bipartite matching
    matches = greedy_match(iou_mat)
    
    matched_pred_indices = set()
    matched_gt_indices = set()
    total_score = 0.0
    
    for pi, gj in matches:
        matched_pred_indices.add(pi)
        matched_gt_indices.add(gj)
        iou_val = iou_mat[pi, gj].item()
        # Base IoU score
        pair_score = iou_val
        # Bonus for high-quality matches
        if iou_val > iou_bonus_threshold:
            pair_score += iou_bonus_value
        total_score += pair_score
    
    # False positive penalty (unmatched predictions)
    num_fp = M - len(matched_pred_indices)
    total_score += fp_penalty * num_fp
    
    # False negative penalty (unmatched GT)
    num_fn = N - len(matched_gt_indices)
    total_score += fn_penalty * num_fn
    
    # Normalize by number of GT targets
    reward = total_score / max(1, N)
    
    # return reward
    return max(-3.0, reward)


def compute_format_reward(text: str, seg_token: str = "[SEG]") -> float:
    """
    Compute format compliance reward for CoT structure.
    
    Includes consistency keyword bonus for dual-stream acknowledgement.
    
    Args:
        text: Generated text
        seg_token: Segmentation token marker
    
    Returns:
        Format reward in [-1.0, 1.0]
    """
    reward = 0.0
    
    # Critical: must have [SEG] token
    if seg_token not in text:
        return -1.0
    
    # =========================================================================
    # Strict <p> tag syntax checks
    # These fire BEFORE any positive bonus to ensure structural compliance
    #
    # 正确格式:  <p> region_desc </p> [SEG] <p> region_desc </p> [SEG] ...
    # 即每个 segment 都是  <p> ... </p> [SEG]  的完整单元
    # =========================================================================
    p_start_count = text.count("<p>")
    p_end_count = text.count("</p>")
    seg_count = text.count(seg_token)
    
    # 1. <p> 和 </p> 必须严格成对 (不闭合 = 结构性灾难)
    if p_start_count != p_end_count:
        reward -= 2.0
        
    # 2. </p> 数量必须等于 [SEG] 数量 (裸 SEG 或多余标签 = 乱吐)
    if p_end_count != seg_count:
        reward -= 2.0
        
    # # 3. 输出必须以 <p> 开头 (新数据格式要求)
    # if not text.lstrip().startswith("<p>"):
    #     reward -= 1.0
    
    # 4. 检查 <p>...</p> [SEG] 的正确交替顺序
    #    用正则提取所有合法的 "<p> ... </p> [SEG]" 单元
    #    合法单元要求: <p> 内必须有非空内容, 且紧跟 [SEG]
    #    预期格式(单伪迹):   <p>desc</p>[SEG] Step1... Step2... Step3...
    #    预期格式(多伪迹):   <p>desc1</p>[SEG] Step1..2..3.. <p>desc2</p>[SEG] Step1..2..3..
    #    如果合法单元数 < seg_count, 说明有顺序错乱或内容为空
    #    (例如 "</p> </p> [SEG] [SEG]" 数量对但顺序不对)
    valid_units = re.findall(
        r'<p>.+?</p>\s*' + re.escape(seg_token),  # .+? 要求 <p> 内至少有一个字符
        text,
        re.DOTALL
    )
    if len(valid_units) < seg_count:
        # 按差值比例罚: 差得越多罚得越重
        reward -= 1.0 * (seg_count - len(valid_units))

    # 4b. 显式惩罚空 <p></p> 标签 (内容为空或纯空白)
    empty_p_count = len(re.findall(r'<p>\s*</p>', text))
    if empty_p_count > 0:
        reward -= 1.0 * empty_p_count
    # =========================================================================
    
    # =========================================================================
    # 5. Gibberish / non-English penalty
    #    Strip out allowed markup, then check remaining text is clean English.
    #    "Clean" = ASCII letters, digits, spaces, and common punctuation.
    #    If clean ratio < 70%, model is likely spewing gibberish.
    # =========================================================================
    stripped = text
    for token_to_remove in ["<p>", "</p>", seg_token]:
        stripped = stripped.replace(token_to_remove, "")
    stripped = stripped.strip()
    if len(stripped) > 0:
        # Allow: a-z A-Z 0-9, spaces, and common English punctuation
        clean_chars = sum(
            1 for c in stripped
            if c.isascii() and (c.isalnum() or c in ' .,;:!?\'"-()/')
        )
        clean_ratio = clean_chars / len(stripped)
        if clean_ratio < 0.7:
            reward -= 1.5  # Heavy penalty for gibberish
    # =========================================================================
    
    reward += 0.3
    
    # Check for step structure
    step_patterns = [
        r"Step\s*\d+",
        r"^\s*\d+\.",
        r"(?:First|Second|Third|Finally)",
        r"(?:首先|其次|然后|最后)",
    ]
    if any(re.search(p, text, re.MULTILINE | re.IGNORECASE) for p in step_patterns):
        reward += 0.2
    
    # =========================================================================
    # DUAL-STREAM: DIRE map keyword bonus for Step 2
    # Based on SynthScars_CoT annotation patterns, Step 2 describes DIRE map
    # =========================================================================
    dire_keywords = [
        # Core DIRE terminology
        r"DIRE\s*map",
        r"reconstruction\s*error",
        r"reconstruction\s*failure",
        r"reconstruction\s*difficulty",
        # Visual patterns in DIRE
        r"bright\s*region",
        r"hotspot",
        r"high.?intensity",
        r"gray.?white\s*noise",
        r"noise\s*pattern",
        r"elevated.*error",
        r"high.*error",
        # Structural descriptions
        r"structural\s*inconsistency",
        r"structural\s*loss",
        r"geometry",
        r"diffuse",
        r"concentrated",
        r"localized",
    ]
    
    # Look for Step 2 content specifically
    step2_match = re.search(
        r"(?:Step\s*2|Second)[:\s]*(.{20,500})",
        text, 
        re.IGNORECASE | re.DOTALL
    )
    
    if step2_match:
        step2_content = step2_match.group(1).lower()
        # Check if Step 2 mentions DIRE-related concepts
        dire_mentions = sum(
            1 for kw in dire_keywords 
            if re.search(kw, step2_content, re.IGNORECASE)
        )
        if dire_mentions >= 1:
            reward += 0.15  # Bonus for acknowledging DIRE stream
        if dire_mentions >= 3:
            reward += 0.1   # Extra bonus for detailed DIRE description
    
    # Check for conclusion
    conclusion_patterns = [
        r"(?:conclusion|therefore|thus|in summary)",
        # r"(?:结论|因此|综上)",
        r"(?:forged|manipulated|tampered|authentic|real|fake)",
    ]
    if any(re.search(p, text, re.IGNORECASE) for p in conclusion_patterns):
        reward += 0.15
    
    # Length check
    if len(text.split()) >= 20:
        reward += 0.05
    elif len(text.split()) < 5:
        reward -= 0.3
    
    # Repetition penalty
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    if len(sentences) > 3:
        unique_ratio = len(set(sentences)) / len(sentences)
        if unique_ratio < 0.5:
            reward -= 0.3
            
    # =========================================================================
    # Penalty for excessive [SEG] token repetition
    # Normal outputs have 1-3 [SEG] tokens; more indicates looping behavior
    # =========================================================================
    seg_count = text.count(seg_token)
    
    # Strict penalty for excessive [SEG] tokens (>3).
    # No cap — spamming [SEG] pays an uncapped linear cost here,
    # AND each extra pred also gets a FP penalty in multi-mask matching.
    if seg_count > 3:
        excess = seg_count - 3
        reward -= 0.3 * excess  # e.g. 6 SEGs → -0.9, 10 SEGs → -2.1
    
    return max(-3.0, min(1.0, reward))


def compute_mask_reward(
    pred_mask: Optional[torch.Tensor],
    gt_mask: Optional[torch.Tensor],
    iou_bonus_threshold: float = 0.7,
    iou_bonus_value: float = 0.5,
) -> float:
    """
    Compute mask quality reward based on IoU.
    
    Reward logic:
    - Base = IoU score
    - Both empty → 1.0
    - Bonus +0.5 if IoU > 0.7
    """
    # Helper to check if mask is empty
    def is_empty(mask):
        if mask is None:
            return True
        if isinstance(mask, (list, tuple)):
            return len(mask) == 0 or all(is_empty(m) for m in mask)
        if isinstance(mask, torch.Tensor):
            return mask.numel() == 0 or mask.sum() == 0
        # numpy array or other
        try:
            import numpy as np
            if isinstance(mask, np.ndarray):
                return mask.size == 0 or mask.sum() == 0
        except:
            pass
        return False
    
    pred_empty = is_empty(pred_mask)
    gt_empty = is_empty(gt_mask)
    
    if pred_empty and gt_empty:
        return 1.0
    if pred_empty != gt_empty:
        return 0.0
    
    iou = compute_iou(pred_mask, gt_mask)
    reward = iou
    if iou > iou_bonus_threshold:
        reward += iou_bonus_value
    
    return reward


# ============================================================================
# LaPReward Class - The Core Reward Module
# ============================================================================

class LaPReward:
    """
    Class-based reward module for TRL GRPO training.
    
    Holds references to dataset, model, and tokenizer to enable:
    1. Looking up gt_masks and images using idx from kwargs
    2. Running forward pass to extract [SEG] hidden states
    3. Generating masks via SAM decoder
    4. Computing IoU-based rewards
    
    Usage:
        reward_module = LaPReward(dataset, model, tokenizer)
        trainer = GRPOTrainer(..., reward_funcs=reward_module)
    """
    
    def __init__(
        self,
        dataset,
        model,
        tokenizer,
        mask_reward_weight: float = 1.0,
        format_reward_weight: float = 0.3,
        seg_token: str = "[SEG]",
        iou_bonus_threshold: float = 0.7,
        iou_bonus_value: float = 0.5,
        missing_seg_penalty: float = -2.0,
    ):
        """
        Initialize the reward module.
        
        Args:
            dataset: TRLDatasetWrapper instance (same as used by Trainer)
            model: LaPForCausalLM model (with LoRA applied)
            tokenizer: Tokenizer instance
            mask_reward_weight: Weight for mask IoU reward
            format_reward_weight: Weight for format compliance reward
            seg_token: The [SEG] token string
            iou_bonus_threshold: IoU threshold for bonus reward
            iou_bonus_value: Bonus value when IoU exceeds threshold
            missing_seg_penalty: Heavy penalty when [SEG] token is missing (default -2.0)
        """
        self.dataset = dataset
        self.model = model
        self.tokenizer = tokenizer
        self.mask_reward_weight = mask_reward_weight
        self.format_reward_weight = format_reward_weight
        self.seg_token = seg_token
        self.seg_token_id = tokenizer.convert_tokens_to_ids(seg_token)
        self.iou_bonus_threshold = iou_bonus_threshold
        self.iou_bonus_value = iou_bonus_value
        self.missing_seg_penalty = missing_seg_penalty
        
        # Track statistics
        self.call_count = 0
        self._debug_last_log = 0
        self.debug = os.getenv("LAP_FORENSIC_DEBUG", "0") != "0"
        # Skip mask reward for faster debugging (only use format reward)
        self.skip_mask_reward = os.getenv("LAP_FORENSIC_SKIP_MASK_REWARD", "0") != "0"
        try:
            self.debug_every = int(os.getenv("LAP_FORENSIC_DEBUG_EVERY", "10"))
        except ValueError:
            self.debug_every = 10
        
        # TRL GRPOTrainer expects reward functions to have __name__ attribute
        self.__name__ = "LaPReward"
        
        # Batched reward computation (reduces GPU idle time)
        # Set LAP_FORENSIC_BATCHED_REWARD=0 to disable
        self.use_batched_reward = os.getenv("LAP_FORENSIC_BATCHED_REWARD", "1") != "0"
        
        logger.info(f"LaPReward initialized: missing_seg_penalty={missing_seg_penalty}, batched={self.use_batched_reward}")
    
    def _unwrap_model_for_forward(self):
        """
        Unwrap model to LaPForCausalLM level for forward pass.
        This level has prepare_inputs_labels_for_multimodal for image processing.
        """
        model = self.model
        
        # Unwrap DDP
        if hasattr(model, 'module'):
            model = model.module
        
        # Unwrap PEFT to get LaPForCausalLM (but NOT deeper!)
        if hasattr(model, 'base_model'):
            # For PEFT, base_model.model is usually the actual model
            if hasattr(model.base_model, 'model'):
                model = model.base_model.model
            else:
                model = model.base_model
            
        return model
    
    def _unwrap_model_for_internals(self):
        """
        Unwrap model to LaPModel level to access text_hidden_fcs and grounding_encoder.
        """
        model = self._unwrap_model_for_forward()
        
        # Go one level deeper to get LaPModel (which has text_hidden_fcs)
        if hasattr(model, 'model'):
            return model.model
            
        return model
    
    def _get_sample_data(self, idx: int) -> Dict[str, Any]:
        """
        Retrieve sample data from dataset using index.
        
        Args:
            idx: Sample index
            
        Returns:
            Dict containing gt_mask, grounding_enc_image, resize, label
        """
        try:
            # Access the base dataset through wrapper
            if hasattr(self.dataset, '_metadata_cache'):
                # Use cached metadata if available
                metadata = self.dataset._metadata_cache.get(idx, None)
                if metadata:
                    return {
                        'gt_mask': metadata.get('masks'),
                        'grounding_enc_image': metadata.get('grounding_enc_image'),
                        'global_enc_image': metadata.get('global_enc_image'),  # For forward pass
                        'resize': metadata.get('resize'),
                        'label': metadata.get('label'),
                    }
            
            # Fallback: directly access base dataset
            if hasattr(self.dataset, 'base_dataset'):
                raw_data = self.dataset.base_dataset[idx]
                # Unpack LaPGCGDataset format
                (image_path, global_enc_image, grounding_enc_image, bboxes,
                 conversations, masks, label, resize, questions, sampled_classes) = raw_data
                
                return {
                    'gt_mask': masks,
                    'grounding_enc_image': grounding_enc_image,
                    'global_enc_image': global_enc_image,  # For forward pass
                    'resize': resize,
                    'label': label,
                }
        except Exception:
            pass
        
        return {
            'gt_mask': None,
            'grounding_enc_image': None,
            'global_enc_image': None,
            'resize': None,
            'label': None,
        }
    
    def _extract_seg_hidden_state(
        self,
        prompt: str,
        completion: str,
        device: torch.device,
        global_enc_image: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """
        Extract hidden states at ALL [SEG] token positions.
        
        This runs a forward pass on prompt + completion WITH image context
        to get the hidden states that will be used for mask generation.
        
        Args:
            prompt: The input prompt string
            completion: The generated completion string
            device: Device to run on
            global_enc_image: CLIP-encoded image for visual context
            
        Returns:
            List of hidden state tensors, each [1, hidden_dim].
            Empty list if no [SEG] found.
        """
        # Concatenate prompt + completion
        full_text = prompt + completion
        
        # Debug: check if <image> is in prompt
        if self.debug and self.call_count % self.debug_every == 1:
            has_image_token = "<image>" in full_text
            has_seg_token = self.seg_token in full_text
            logger.info("SEG extract debug: has_image=%s, has_seg=%s, text_len=%d", 
                       has_image_token, has_seg_token, len(full_text))
        
        # Tokenize
        inputs = self.tokenizer(
            full_text, 
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )
        input_ids = inputs.input_ids.to(device)
        
        # CRITICAL: Replace <image> tokens with IMAGE_TOKEN_INDEX (-200)
        # Tokenizer converts "<image>" to regular tokens, but model expects -200
        IMAGE_TOKEN_INDEX = -200
        image_str_ids = self.tokenizer("<image>", add_special_tokens=False).input_ids
        
        # Replace the sequence in input_ids
        input_ids_list = input_ids[0].tolist()
        new_ids = []
        i = 0
        while i < len(input_ids_list):
            if i + len(image_str_ids) <= len(input_ids_list):
                if input_ids_list[i:i+len(image_str_ids)] == image_str_ids:
                    new_ids.append(IMAGE_TOKEN_INDEX)
                    i += len(image_str_ids)
                    continue
            new_ids.append(input_ids_list[i])
            i += 1
        
        input_ids = torch.tensor([new_ids], device=device, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        
        # Find ALL [SEG] token positions in input_ids
        seg_positions = (input_ids[0] == self.seg_token_id).nonzero(as_tuple=True)[0]
        if len(seg_positions) == 0:
            return []
        
        # Check if [SEG]s are after image token (needs offset adjustment)
        image_positions = (input_ids[0] == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0]
        has_image = len(image_positions) > 0
        image_pos = image_positions[0].item() if has_image else -1
        
        # Prepare image if available
        images = None
        if global_enc_image is not None:
            images = global_enc_image.to(device)
            if images.dim() == 3:
                images = images.unsqueeze(0)
            model_dtype = next(self.model.parameters()).dtype
            images = images.to(dtype=model_dtype)
        
        # Get LaPForCausalLM for forward pass
        forward_model = self._unwrap_model_for_forward()
        
        with torch.no_grad():
            outputs = forward_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                images=images,
                output_hidden_states=True,
                return_dict=True,
            )
            
            # Get last layer hidden states
            hidden_states = outputs.hidden_states
            if isinstance(hidden_states, (list, tuple)):
                hidden_states = hidden_states[-1]
            
            # Handle 2D case: [seq_len, hidden_dim] -> [1, seq_len, hidden_dim]
            if hidden_states.dim() == 2:
                hidden_states = hidden_states.unsqueeze(0)
            
            # Calculate image patch offset for position adjustment
            hidden_len = hidden_states.shape[1]
            input_len = len(new_ids)
            num_image_patches = hidden_len - input_len + 1
            
            # Extract hidden state at EACH [SEG] position
            seg_hiddens = []
            for seg_pos_raw in seg_positions:
                seg_pos = seg_pos_raw.item()
                
                # Apply offset if this [SEG] comes after the image token
                need_offset = has_image and (seg_pos > image_pos) and (num_image_patches > 0)
                if need_offset:
                    actual_pos = seg_pos + num_image_patches - 1
                else:
                    actual_pos = seg_pos
                
                # Clamp position
                if actual_pos >= hidden_len:
                    actual_pos = hidden_len - 1
                
                seg_hidden = hidden_states[0, actual_pos:actual_pos+1, :]
                seg_hiddens.append(seg_hidden)
            
            return seg_hiddens
    
    def _generate_mask(
        self,
        seg_hidden: torch.Tensor,
        grounding_enc_image: torch.Tensor,
        resize_info: Optional[Tuple[int, int]],
        label: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """
        Generate segmentation mask from [SEG] hidden state using SAM decoder.
        
        Args:
            seg_hidden: Hidden state at [SEG] position [1, hidden_dim]
            grounding_enc_image: SAM-preprocessed image [1, 3, H, W]
            resize_info: Tuple of (resized_h, resized_w)
            label: Ground truth label for original size reference
            
        Returns:
            Predicted mask tensor [H, W] or None
        """
        # Get LaPModel level (has text_hidden_fcs and grounding_encoder)
        lap_model = self._unwrap_model_for_internals()
        
        if not hasattr(lap_model, 'text_hidden_fcs'):
            raise AttributeError(f"Cannot find text_hidden_fcs in model. Type: {type(lap_model)}")
        
        # Ensure image has batch dimension
        if grounding_enc_image.dim() == 3:
            grounding_enc_image = grounding_enc_image.unsqueeze(0)
        
        device = grounding_enc_image.device
        model_dtype = next(lap_model.text_hidden_fcs[0].parameters()).dtype
        
        # Cast tensors to model dtype (bfloat16)
        seg_hidden = seg_hidden.to(device=device, dtype=model_dtype)
        grounding_enc_image = grounding_enc_image.to(dtype=model_dtype)
        
        with torch.no_grad():
            # Project hidden state to SAM embedding space
            pred_embedding = lap_model.text_hidden_fcs[0](seg_hidden)
            
            # Get SAM image embeddings
            image_embeddings = lap_model.grounding_encoder.image_encoder(grounding_enc_image)
            
            # Generate mask via SAM decoder
            sparse_emb, dense_emb = lap_model.grounding_encoder.prompt_encoder(
                points=None, 
                boxes=None, 
                masks=None,
                text_embeds=pred_embedding.unsqueeze(1)
            )
            sparse_emb = sparse_emb.to(pred_embedding.dtype)
            
            low_res_masks, _ = lap_model.grounding_encoder.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=lap_model.grounding_encoder.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=False,
            )
            
            # Postprocess to original size
            if resize_info is not None and label is not None:
                orig_size = label.shape[-2:] if hasattr(label, 'shape') else label
                pred_mask = lap_model.grounding_encoder.postprocess_masks(
                    low_res_masks,
                    input_size=resize_info,
                    original_size=orig_size,
                )
                return pred_mask[0, 0]
            
            return low_res_masks[0, 0]
    
    def _generate_masks_batched(
        self,
        seg_hiddens: List[torch.Tensor],
        grounding_enc_images: List[torch.Tensor],
        resize_infos: List[Optional[Tuple[int, int]]],
        labels: List[Optional[torch.Tensor]],
    ) -> List[Optional[torch.Tensor]]:
        """
        Batched mask generation using SAM decoder.
        
        Processes multiple seg_hidden states through SAM in parallel batches
        for improved GPU utilization.
        
        Args:
            seg_hiddens: List of [1, hidden_dim] tensors
            grounding_enc_images: List of [3, H, W] or [1, 3, H, W] tensors
            resize_infos: List of resize info tuples
            labels: List of label tensors for original size
            
        Returns:
            List of predicted mask tensors (or None for failed samples)
        """
        if not seg_hiddens:
            return []
        
        # Filter out None entries and track indices
        valid_indices = []
        valid_seg_hiddens = []
        valid_grounding_images = []
        valid_resize_infos = []
        valid_labels = []
        
        for i, (seg_h, img, resize, label) in enumerate(
            zip(seg_hiddens, grounding_enc_images, resize_infos, labels)
        ):
            if seg_h is not None and img is not None:
                valid_indices.append(i)
                valid_seg_hiddens.append(seg_h)
                valid_grounding_images.append(img)
                valid_resize_infos.append(resize)
                valid_labels.append(label)
        
        if not valid_seg_hiddens:
            return [None] * len(seg_hiddens)
        
        # Get LaPModel level
        lap_model = self._unwrap_model_for_internals()
        
        if not hasattr(lap_model, 'text_hidden_fcs'):
            logger.warning("Cannot find text_hidden_fcs in model for batched mask gen")
            return [None] * len(seg_hiddens)
        
        device = valid_grounding_images[0].device
        model_dtype = next(lap_model.text_hidden_fcs[0].parameters()).dtype
        
        # Process in mini-batches to avoid OOM (batch size of 16)
        batch_size = 16
        all_pred_masks = [None] * len(seg_hiddens)
        
        with torch.no_grad():
            for batch_start in range(0, len(valid_seg_hiddens), batch_size):
                batch_end = min(batch_start + batch_size, len(valid_seg_hiddens))
                batch_seg_hiddens = valid_seg_hiddens[batch_start:batch_end]
                batch_images = valid_grounding_images[batch_start:batch_end]
                batch_resize = valid_resize_infos[batch_start:batch_end]
                batch_labels = valid_labels[batch_start:batch_end]
                batch_indices = valid_indices[batch_start:batch_end]
                
                # Stack seg_hiddens: [B, 1, hidden_dim]
                stacked_seg = torch.cat(batch_seg_hiddens, dim=0).to(device=device, dtype=model_dtype)
                
                # Stack images: [B, 3, H, W]
                stacked_images = []
                for img in batch_images:
                    if img.dim() == 3:
                        img = img.unsqueeze(0)
                    stacked_images.append(img.to(dtype=model_dtype))
                stacked_images = torch.cat(stacked_images, dim=0)
                
                # Project hidden states to SAM embedding space: [B, hidden_dim] -> [B, embedding_dim]
                pred_embeddings = lap_model.text_hidden_fcs[0](stacked_seg)
                
                # Get SAM image embeddings for all images in batch
                image_embeddings = lap_model.grounding_encoder.image_encoder(stacked_images)
                
                # Generate masks for each sample (SAM doesn't support true batch with different prompts)
                for j, (idx, pred_emb, img_emb, resize, label) in enumerate(
                    zip(batch_indices, pred_embeddings, image_embeddings, batch_resize, batch_labels)
                ):
                    try:
                        sparse_emb, dense_emb = lap_model.grounding_encoder.prompt_encoder(
                            points=None,
                            boxes=None,
                            masks=None,
                            text_embeds=pred_emb.unsqueeze(0).unsqueeze(1)  # [1, 1, embedding_dim]
                        )
                        sparse_emb = sparse_emb.to(pred_emb.dtype)
                        
                        low_res_masks, _ = lap_model.grounding_encoder.mask_decoder(
                            image_embeddings=img_emb.unsqueeze(0),  # [1, C, H, W]
                            image_pe=lap_model.grounding_encoder.prompt_encoder.get_dense_pe(),
                            sparse_prompt_embeddings=sparse_emb,
                            dense_prompt_embeddings=dense_emb,
                            multimask_output=False,
                        )
                        
                        # Postprocess to original size
                        if resize is not None and label is not None:
                            orig_size = label.shape[-2:] if hasattr(label, 'shape') else label
                            pred_mask = lap_model.grounding_encoder.postprocess_masks(
                                low_res_masks,
                                input_size=resize,
                                original_size=orig_size,
                            )
                            all_pred_masks[idx] = pred_mask[0, 0]
                        else:
                            all_pred_masks[idx] = low_res_masks[0, 0]
                    except Exception as e:
                        if self.debug:
                            logger.warning("Batched mask gen failed for idx %d: %s", idx, str(e)[:50])
                        all_pred_masks[idx] = None
        
        return all_pred_masks
    
    def _call_batched(
        self,
        prompts: List[str],
        completions: List[str],
        indices: List[Optional[int]],
        device: torch.device,
        call_start: float,
    ) -> List[float]:
        """
        Batched reward computation for improved GPU utilization.
        
        Key optimizations:
        1. Prefetch sample_data for all unique indices
        2. Batch SAM image encoding across samples
        3. Process seg_hidden extraction in larger groups
        """
        n = len(completions)
        rewards = [0.0] * n
        format_rewards = [0.0] * n
        mask_rewards = [0.0] * n
        
        # Track which completions need mask computation
        need_mask_compute = [False] * n
        
        # Phase 1: Compute format rewards and identify valid completions
        missing_seg_count = 0
        for i, completion in enumerate(completions):
            has_seg = self.seg_token in completion
            if not has_seg:
                missing_seg_count += 1
                format_rewards[i] = -1.0
                rewards[i] = self.missing_seg_penalty
            else:
                format_rewards[i] = compute_format_reward(completion, self.seg_token)
                need_mask_compute[i] = True
        
        # Phase 2: Prefetch sample_data for all unique indices
        unique_indices = set(idx for idx in indices if idx is not None)
        sample_data_cache = {}
        
        sample_data_start = time.perf_counter()
        for idx in unique_indices:
            sample_data_cache[idx] = self._get_sample_data(idx)
        sample_data_time = time.perf_counter() - sample_data_start
        
        # Phase 3: Group completions by sample index for batched image encoding
        # This allows us to encode each unique grounding_enc_image only once
        idx_to_completions = {}  # idx -> [(i, prompt, completion), ...]
        for i, (prompt, completion) in enumerate(zip(prompts, completions)):
            if not need_mask_compute[i]:
                continue
            idx = indices[i] if i < len(indices) else None
            if idx is None:
                continue
            if idx not in idx_to_completions:
                idx_to_completions[idx] = []
            idx_to_completions[idx].append((i, prompt, completion))
        
        # Phase 4: Process each sample's completions
        # Batch the SAM image encoding across samples
        seg_extract_time = 0.0
        mask_gen_time = 0.0
        iou_time = 0.0
        exception_count = 0
        
        # Pre-encode all unique grounding images
        lap_model = self._unwrap_model_for_internals()
        model_dtype = next(self.model.parameters()).dtype
        
        idx_to_image_embedding = {}
        grounding_images_to_encode = []
        grounding_idx_order = []
        
        for idx, sample_data in sample_data_cache.items():
            grounding_enc_image = sample_data.get('grounding_enc_image')
            if grounding_enc_image is not None and idx in idx_to_completions:
                grounding_images_to_encode.append(grounding_enc_image)
                grounding_idx_order.append(idx)
        
        # Batch encode SAM images (significant speedup)
        if grounding_images_to_encode and hasattr(lap_model, 'grounding_encoder'):
            try:
                mask_gen_start = time.perf_counter()
                with torch.no_grad():
                    # Stack images into batch
                    stacked_images = []
                    for img in grounding_images_to_encode:
                        if img.dim() == 3:
                            img = img.unsqueeze(0)
                        stacked_images.append(img.to(device=device, dtype=model_dtype))
                    
                    # Encode in mini-batches of 8 to avoid OOM
                    batch_size = 8
                    all_embeddings = []
                    for batch_start in range(0, len(stacked_images), batch_size):
                        batch_end = min(batch_start + batch_size, len(stacked_images))
                        batch_images = torch.cat(stacked_images[batch_start:batch_end], dim=0)
                        batch_embeddings = lap_model.grounding_encoder.image_encoder(batch_images)
                        all_embeddings.append(batch_embeddings)
                    
                    # Combine and map to indices
                    combined_embeddings = torch.cat(all_embeddings, dim=0)
                    for i, idx in enumerate(grounding_idx_order):
                        idx_to_image_embedding[idx] = combined_embeddings[i:i+1]
                
                mask_gen_time += time.perf_counter() - mask_gen_start
            except Exception as e:
                logger.warning("Batched image encoding failed: %s", str(e)[:100])
                # Fallback: will encode individually
        
        # Phase 5: Process each completion's mask reward
        for idx, completion_list in idx_to_completions.items():
            sample_data = sample_data_cache.get(idx)
            if sample_data is None:
                continue
            
            gt_mask = sample_data.get('gt_mask')
            grounding_enc_image = sample_data.get('grounding_enc_image')
            global_enc_image = sample_data.get('global_enc_image')
            resize_info = sample_data.get('resize')
            label = sample_data.get('label')
            
            if gt_mask is None or grounding_enc_image is None:
                continue
            
            # Prepare individual GT masks (NO merging)
            gt_masks_list = self._prepare_gt_masks_list(gt_mask, device)
            
            # Get pre-encoded image embedding if available
            image_embedding = idx_to_image_embedding.get(idx)
            
            # Process each completion for this sample
            for i, prompt, completion in completion_list:
                try:
                    # Extract ALL seg hidden states
                    seg_start = time.perf_counter()
                    seg_hiddens = self._extract_seg_hidden_state(
                        prompt, completion, device,
                        global_enc_image=global_enc_image,
                    )
                    seg_extract_time += time.perf_counter() - seg_start
                    
                    if not seg_hiddens:
                        continue
                    
                    # Generate one mask per SEG hidden state
                    mask_start = time.perf_counter()
                    pred_masks = self._generate_masks_from_hiddens(
                        seg_hiddens, image_embedding, grounding_enc_image,
                        resize_info, label, device,
                    )
                    mask_gen_time += time.perf_counter() - mask_start
                    
                    # Multi-mask matching reward
                    iou_start = time.perf_counter()
                    r_mask = compute_multi_mask_reward(
                        pred_masks, gt_masks_list,
                        self.iou_bonus_threshold, self.iou_bonus_value,
                    )
                    iou_time += time.perf_counter() - iou_start
                    
                    mask_rewards[i] = r_mask
                    
                except Exception as e:
                    exception_count += 1
                    if self.debug:
                        logger.warning("Batched reward exception at %d: %s", i, str(e)[:50])
        
        # Phase 6: Combine rewards
        for i in range(n):
            if not need_mask_compute[i]:
                continue  # Already set (missing SEG penalty)
            total = self.mask_reward_weight * mask_rewards[i] + self.format_reward_weight * format_rewards[i]
            rewards[i] = total
        
        # Store stats
        self.last_stats = {
            'reward_mean': sum(rewards) / len(rewards) if rewards else 0,
            'reward_min': min(rewards) if rewards else 0,
            'reward_max': max(rewards) if rewards else 0,
            'format_reward_mean': sum(format_rewards) / len(format_rewards) if format_rewards else 0,
            'mask_reward_mean': sum(mask_rewards) / len(mask_rewards) if mask_rewards else 0,
        }
        
        # Debug logging
        if self.debug and self.call_count % self.debug_every == 1:
            call_time = time.perf_counter() - call_start
            logger.info(
                "Batched reward: call=%d, completions=%d, total=%.2fs, "
                "sample_data=%.2fs, seg_extract=%.2fs, mask_gen=%.2fs, iou=%.2fs, "
                "missing_seg=%d, exceptions=%d, unique_samples=%d",
                self.call_count, n, call_time,
                sample_data_time, seg_extract_time, mask_gen_time, iou_time,
                missing_seg_count, exception_count, len(unique_indices),
            )
        
        return rewards
    
    def _generate_mask_with_embedding(
        self,
        seg_hidden: torch.Tensor,
        image_embedding: torch.Tensor,
        resize_info: Optional[Tuple[int, int]],
        label: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """
        Generate mask using pre-computed SAM image embedding.
        
        This avoids redundant image encoding for samples shared across completions.
        """
        lap_model = self._unwrap_model_for_internals()
        model_dtype = next(lap_model.text_hidden_fcs[0].parameters()).dtype
        device = image_embedding.device
        
        seg_hidden = seg_hidden.to(device=device, dtype=model_dtype)
        
        with torch.no_grad():
            # Project hidden state
            pred_embedding = lap_model.text_hidden_fcs[0](seg_hidden)
            
            # Generate mask via SAM decoder
            sparse_emb, dense_emb = lap_model.grounding_encoder.prompt_encoder(
                points=None, boxes=None, masks=None,
                text_embeds=pred_embedding.unsqueeze(1)
            )
            sparse_emb = sparse_emb.to(pred_embedding.dtype)
            
            low_res_masks, _ = lap_model.grounding_encoder.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=lap_model.grounding_encoder.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=False,
            )
            
            # Postprocess
            if resize_info is not None and label is not None:
                orig_size = label.shape[-2:] if hasattr(label, 'shape') else label
                pred_mask = lap_model.grounding_encoder.postprocess_masks(
                    low_res_masks, input_size=resize_info, original_size=orig_size
                )
                return pred_mask[0, 0]
            
            return low_res_masks[0, 0]
    
    def _prepare_gt_masks_list(
        self,
        gt_mask,
        device: torch.device,
    ) -> List[torch.Tensor]:
        """
        Convert gt_mask (Tensor [N, H, W], list, or [H, W]) into a list of
        individual 2D mask tensors. Does NOT merge masks.
        """
        if gt_mask is None:
            return []
        
        if isinstance(gt_mask, torch.Tensor):
            gt_mask = gt_mask.to(device)
            if gt_mask.dim() == 3:
                return [gt_mask[k] for k in range(gt_mask.shape[0])]
            elif gt_mask.dim() == 2:
                return [gt_mask]
            else:
                # Squeeze extra dims
                while gt_mask.dim() > 2:
                    gt_mask = gt_mask.squeeze(0)
                return [gt_mask]
        elif isinstance(gt_mask, (list, tuple)):
            result = []
            for m in gt_mask:
                if not isinstance(m, torch.Tensor):
                    m = torch.tensor(m, device=device)
                else:
                    m = m.to(device)
                while m.dim() > 2:
                    m = m.squeeze(0)
                result.append(m)
            return result
        else:
            return []
    
    def _generate_masks_from_hiddens(
        self,
        seg_hiddens: List[torch.Tensor],
        image_embedding: Optional[torch.Tensor],
        grounding_enc_image: Optional[torch.Tensor],
        resize_info: Optional[Tuple[int, int]],
        label: Optional[torch.Tensor],
        device: torch.device,
    ) -> List[Optional[torch.Tensor]]:
        """
        Generate one predicted mask per SEG hidden state.
        
        Uses pre-computed image_embedding if available, otherwise falls back
        to full grounding_enc_image encoding.
        
        Args:
            seg_hiddens: List of [1, hidden_dim] tensors
            image_embedding: Pre-computed SAM image embedding (or None)
            grounding_enc_image: Raw SAM-preprocessed image (fallback)
            resize_info: Resize info tuple
            label: Label for original size
            device: Torch device
            
        Returns:
            List of predicted mask tensors [H, W] (or None for failures)
        """
        pred_masks = []
        for seg_h in seg_hiddens:
            try:
                if image_embedding is not None:
                    mask = self._generate_mask_with_embedding(
                        seg_h, image_embedding, resize_info, label
                    )
                elif grounding_enc_image is not None:
                    mask = self._generate_mask(
                        seg_h, grounding_enc_image.to(device), resize_info, label
                    )
                else:
                    mask = None
                pred_masks.append(mask)
            except Exception as e:
                if self.debug:
                    logger.warning("Mask gen failed for seg_hidden: %s", str(e)[:50])
                pred_masks.append(None)
        return pred_masks
    
    def __call__(
        self,
        prompts: List[str],
        completions: List[str],
        **kwargs
    ) -> List[float]:
        """
        Compute rewards for a batch of completions.
        
        This is the main entry point called by TRL GRPOTrainer.
        
        Args:
            prompts: List of prompt strings
            completions: List of completion strings
            **kwargs: Additional context from TRL (includes 'idx')
            
        Returns:
            List of reward values (one per completion)
        """
        self.call_count += 1
        call_start = time.perf_counter()
        
        # Get sample indices from TRL kwargs
        sample_indices = kwargs.get('idx', None)
        indices = self._parse_indices(sample_indices, len(completions), len(prompts))
        
        device = next(self.model.parameters()).device
        
        # Use batched processing for improved GPU utilization
        if self.use_batched_reward and not self.skip_mask_reward:
            return self._call_batched(prompts, completions, indices, device, call_start)
        
        # Original sequential processing (fallback)
        rewards = []
        format_rewards = []
        mask_rewards = []
        missing_seg_count = 0
        exception_count = 0
        cache_hit_count = 0
        cache_miss_count = 0
        sample_data_time = 0.0
        seg_extract_time = 0.0
        mask_gen_time = 0.0
        iou_time = 0.0
        prompt_lens = []
        completion_lens = []
        
        for i, (prompt, completion) in enumerate(zip(prompts, completions)):
            if self.debug and self.call_count % self.debug_every == 1:
                prompt_lens.append(len(prompt))
                completion_lens.append(len(completion))
            
            # CRITICAL: Check if [SEG] token is present in completion
            has_seg_in_completion = self.seg_token in completion
            
            # If [SEG] is missing, apply heavy penalty and skip SAM forward entirely
            if not has_seg_in_completion:
                missing_seg_count += 1
                r_format = -1.0  # Format penalty (no SEG)
                r_mask = 0.0     # Force IoU to 0 (can't compute without SEG)
                format_rewards.append(r_format)
                mask_rewards.append(r_mask)
                # Apply missing_seg_penalty as total reward
                total = self.missing_seg_penalty
                rewards.append(total)
                
                if self.debug and i == 0:
                    logger.info("Missing [SEG] penalty applied: reward=%.2f (completion: '%s')", 
                               total, completion[:100].replace('\n', '\\n'))
                continue
            
            # 1. Compute format reward (SEG is present, so we continue)
            r_format = compute_format_reward(completion, self.seg_token)
            format_rewards.append(r_format)
            
            # 2. Compute mask reward (can be skipped for debugging)
            r_mask = 0.0
            idx = indices[i] if i < len(indices) else None
            
            if self.skip_mask_reward:
                # Skip mask reward computation entirely
                pass
            elif idx is not None:
                if hasattr(self.dataset, "_metadata_cache") and idx in self.dataset._metadata_cache:
                    cache_hit_count += 1
                else:
                    cache_miss_count += 1
                sample_start = time.perf_counter()
                sample_data = self._get_sample_data(idx)
                sample_data_time += time.perf_counter() - sample_start
                gt_mask = sample_data['gt_mask']
                grounding_enc_image = sample_data['grounding_enc_image']
                global_enc_image = sample_data.get('global_enc_image')
                resize_info = sample_data['resize']
                label = sample_data['label']
                
                if gt_mask is not None and grounding_enc_image is not None:
                    try:
                        if self.debug and i == 0:
                            logger.info("Mask reward debug [call %d]: has_seg_in_completion=%s, completion_preview='%s'", 
                                       self.call_count, has_seg_in_completion, completion[:200].replace('\n', '\\n'))
                            print(f"Mask reward debug [call {self.call_count}]: has_seg={has_seg_in_completion}", flush=True)
                        
                        # Extract ALL [SEG] hidden states
                        seg_start = time.perf_counter()
                        seg_hiddens = self._extract_seg_hidden_state(
                            prompt, completion, device,
                            global_enc_image=global_enc_image,
                        )
                        seg_extract_time += time.perf_counter() - seg_start
                        
                        if self.debug and i == 0:
                            seg_info = f"count={len(seg_hiddens)}"
                            logger.info("Mask reward debug: seg_hiddens=%s", seg_info)
                            print(f"Mask reward debug: seg_hiddens={seg_info}", flush=True)
                        
                        if seg_hiddens:
                            # Generate one mask per SEG hidden state
                            grounding_enc_image_dev = grounding_enc_image.to(device)
                            mask_start = time.perf_counter()
                            pred_masks = self._generate_masks_from_hiddens(
                                seg_hiddens, None, grounding_enc_image_dev,
                                resize_info, label, device,
                            )
                            mask_gen_time += time.perf_counter() - mask_start
                            
                            # Prepare individual GT masks (NO merging)
                            gt_masks_list = self._prepare_gt_masks_list(gt_mask, device)
                            
                            iou_start = time.perf_counter()
                            
                            # Debug: Check mask info
                            if self.debug and i == 0:
                                debug_msg = f"num_preds={len(pred_masks)}, num_gt={len(gt_masks_list)}"
                                logger.info("Multi-mask reward debug: %s", debug_msg)
                                print(f"Multi-mask reward debug: {debug_msg}", flush=True)
                            
                            # Multi-mask matching reward
                            r_mask = compute_multi_mask_reward(
                                pred_masks,
                                gt_masks_list,
                                self.iou_bonus_threshold,
                                self.iou_bonus_value,
                            )
                            
                            if self.debug and i == 0:
                                logger.info("Mask reward debug: r_mask=%.6f", r_mask)
                                print(f"Mask reward debug: r_mask={r_mask:.6f}", flush=True)
                            
                            iou_time += time.perf_counter() - iou_start
                        else:
                            # No [SEG] hidden states extracted
                            if self.debug:
                                logger.warning("No seg_hiddens despite [SEG] in completion")
                    except Exception as e:
                        exception_count += 1
                        if self.debug:
                            logger.warning("Mask reward exception: %s", str(e)[:100])
                        r_mask = 0.0
            
            mask_rewards.append(r_mask)
            
            # Combine rewards
            total = self.mask_reward_weight * r_mask + self.format_reward_weight * r_format
            rewards.append(total)
        
        # Store statistics for logging (can be accessed by trainer callback)
        self.last_stats = {
            'reward_mean': sum(rewards) / len(rewards) if rewards else 0,
            'reward_min': min(rewards) if rewards else 0,
            'reward_max': max(rewards) if rewards else 0,
            'format_reward_mean': sum(format_rewards) / len(format_rewards) if format_rewards else 0,
            'mask_reward_mean': sum(mask_rewards) / len(mask_rewards) if mask_rewards else 0,
        }
        
        # Periodic debug logging
        if self.debug and self.call_count % self.debug_every == 1:
            call_time = time.perf_counter() - call_start
            prompt_mean = sum(prompt_lens) / len(prompt_lens) if prompt_lens else 0
            completion_mean = sum(completion_lens) / len(completion_lens) if completion_lens else 0
            logger.info(
                "Reward debug: call=%d, completions=%d, total=%.2fs, "
                "sample_data=%.2fs, seg_extract=%.2fs, mask_gen=%.2fs, iou=%.2fs, "
                "missing_seg=%d, exceptions=%d, cache_hit=%d, cache_miss=%d, "
                "prompt_len_mean=%.1f, completion_len_mean=%.1f",
                self.call_count,
                len(completions),
                call_time,
                sample_data_time,
                seg_extract_time,
                mask_gen_time,
                iou_time,
                missing_seg_count,
                exception_count,
                cache_hit_count,
                cache_miss_count,
                prompt_mean,
                completion_mean,
            )
        
        return rewards
    
    def _parse_indices(
        self, 
        sample_indices: Any, 
        num_completions: int,
        num_prompts: int,
    ) -> List[Optional[int]]:
        """
        Parse sample indices from TRL kwargs into a list.
        
        Handles various formats:
        - None -> list of None
        - int -> [int, int, ...]
        - torch.Tensor -> list
        - List -> expand if needed (GRPO generates multiple completions per prompt)
        
        Args:
            sample_indices: Raw indices from kwargs
            num_completions: Number of completions
            num_prompts: Number of prompts
            
        Returns:
            List of indices (one per completion)
        """
        if sample_indices is None:
            return [None] * num_completions
        
        # Convert to list
        if isinstance(sample_indices, int):
            indices = [sample_indices]
        elif isinstance(sample_indices, torch.Tensor):
            if sample_indices.dim() == 0:
                indices = [sample_indices.item()]
            else:
                indices = sample_indices.tolist()
        else:
            indices = list(sample_indices)
        
        # GRPO generates num_generations completions per prompt
        # Expand indices to match completion count
        if len(indices) < num_completions and len(indices) > 0:
            # Calculate how many completions per sample
            reps = num_completions // len(indices)
            expanded = []
            for idx in indices:
                expanded.extend([idx] * reps)
            # Handle remainder
            while len(expanded) < num_completions:
                expanded.append(indices[-1])
            indices = expanded[:num_completions]
        
        return indices


# ============================================================================
# Factory Function (for backward compatibility)
# ============================================================================

def create_reward_fn(
    model,
    tokenizer,
    dataset=None,
    mask_reward_weight: float = 1.0,
    format_reward_weight: float = 0.3,
    seg_token: str = "[SEG]",
    missing_seg_penalty: float = -2.0,
):
    """
    Create a TRL-compatible reward function.
    
    If dataset is provided, returns a LaPReward instance.
    Otherwise, returns a simple format-only reward function.
    
    Args:
        model: LaPForCausalLM model
        tokenizer: Tokenizer
        dataset: Optional TRLDatasetWrapper for mask rewards
        mask_reward_weight: Weight for mask reward
        format_reward_weight: Weight for format reward
        seg_token: SEG token string
        missing_seg_penalty: Heavy penalty when [SEG] token is missing (default -2.0)
    
    Returns:
        Callable reward function for TRL
    """
    if dataset is not None:
        # Full reward with mask IoU
        return LaPReward(
            dataset=dataset,
            model=model,
            tokenizer=tokenizer,
            mask_reward_weight=mask_reward_weight,
            format_reward_weight=format_reward_weight,
            seg_token=seg_token,
            missing_seg_penalty=missing_seg_penalty,
        )
    else:
        # Format-only reward (fallback)
        def format_only_reward(
            prompts: List[str],
            completions: List[str],
            **kwargs
        ) -> List[float]:
            rewards = []
            for completion in completions:
                r_format = compute_format_reward(completion, seg_token)
                rewards.append(format_reward_weight * r_format)
            return rewards
        
        return format_only_reward
