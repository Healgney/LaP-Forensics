import torch
import torch.nn as nn
from typing import List
import torch.nn.functional as F

from model.SAM import build_sam_vit_h
from model.llava.model.language_model.llava_llama import LlavaLlamaForCausalLM, LlavaLlamaModel
from model.scama_loss import scama_loss
from typing import Optional, List, Tuple, Union
from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass

def calculate_dice_loss(predictions: torch.Tensor, ground_truth: torch.Tensor, mask_count: float, scale_factor=1000,
                        epsilon=1e-6):
    """
    Calculate the DICE loss, a measure similar to generalized IOU for masks.
    """
    predictions = predictions.sigmoid()
    predictions = predictions.flatten(1, 2)
    ground_truth = ground_truth.flatten(1, 2)

    intersection = 2 * (predictions / scale_factor * ground_truth).sum(dim=-1)
    union = (predictions / scale_factor).sum(dim=-1) + (ground_truth / scale_factor).sum(dim=-1)

    dice_loss = 1 - (intersection + epsilon) / (union + epsilon)
    dice_loss = dice_loss.sum() / (mask_count + 1e-8)
    return dice_loss


def compute_sigmoid_cross_entropy(predictions: torch.Tensor, targets: torch.Tensor, mask_count: float):
    """
    Compute sigmoid cross-entropy loss for binary classification.
    """
    loss = F.binary_cross_entropy_with_logits(predictions, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1)
    loss = loss.sum() / (mask_count + 1e-8)
    return loss


class LaPBaseModel:
    def __init__(self, config, **kwargs):
        super(LaPBaseModel, self).__init__(config)
        self.config = config
        self.vision_pretrained = kwargs.get("vision_pretrained", None)

        # Set config attributes if they don't exist
        self.config.train_mask_decoder = getattr(
            self.config, "train_mask_decoder", kwargs.get("train_mask_decoder", False)
        )
        self.config.out_dim = getattr(self.config, "out_dim", kwargs.get("out_dim", 512))

        self.initialize_lap_model(self.config)

    def initialize_lap_model(self, config):
        # Initialize the visual model
        self.grounding_encoder = build_sam_vit_h(self.vision_pretrained)
        self._configure_grounding_encoder(config)

        # Initialize the text projection layer
        self._initialize_text_projection_layer()

    def _configure_grounding_encoder(self, config):
        # Freezing visual model parameters
        for param in self.grounding_encoder.parameters():
            param.requires_grad = False

        # Training mask decoder if specified
        if config.train_mask_decoder:
            self._train_mask_decoder()

    def _train_mask_decoder(self):
        self.grounding_encoder.mask_decoder.train()
        for param in self.grounding_encoder.mask_decoder.parameters():
            param.requires_grad = True

    def _initialize_text_projection_layer(self):
        in_dim, out_dim = self.config.hidden_size, self.config.out_dim
        text_projection_layers = [nn.Linear(in_dim, in_dim), nn.ReLU(inplace=True), nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0), ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_projection_layers)])
        self.text_hidden_fcs.train()
        self.text_hidden_fcs.train()


class LaPModel(LaPBaseModel, LlavaLlamaModel):
    def __init__(self, config, **kwargs):
        super(LaPModel, self).__init__(config, **kwargs)
        self._configure_model_settings()

    def _configure_model_settings(self):
        # NOTE: use_cache should NOT be hardcoded to False here!
        # KV cache is critical for efficient generation in RL training (GRPO).
        # Let it be configurable via the training config instead.
        # self.config.use_cache = False  # DISABLED - causes extremely slow generation
        if not hasattr(self.config, 'use_cache'):
            self.config.use_cache = True  # Default to True for generation efficiency
        self.config.vision_module = self.config.mm_vision_module
        self.config.select_feature_type = "patch"
        self.config.image_aspect = "square"
        self.config.image_grid_points = None
        self.config.tune_mlp_adapter = False
        self.config.freeze_mlp_adapter = True
        self.config.pretrain_mm_mlp_adapter = None
        self.config.use_image_patch_token = False


class LaPForCausalLM(LlavaLlamaForCausalLM):
    def __init__(self, config, **kwargs):
        self._set_model_configurations(config, kwargs)
        super().__init__(config)
        self.model = LaPModel(config, **kwargs)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_parameter_or_buffer(self, target: str):
        try:
            return super().get_parameter_or_buffer(target)
        except AttributeError:
            state_dict = self.state_dict()
            if target in state_dict:
                return state_dict[target]
            raise

    def _set_model_configurations(self, config, kwargs):
        config.mm_use_image_start_end = kwargs.pop("use_mm_start_end", True)
        config.mm_vision_module = kwargs.get("vision_module", "openai/clip-vit-large-patch14-336")
        self._initialize_loss_weights(kwargs)
        config.bbox_token_idx = kwargs.get("bbox_token_idx", 1)
        config.num_reg_features = kwargs.get("num_level_reg_features", 4)
        config.with_region = kwargs.get("with_region", True)
        config.bbox_token_idx = kwargs.get("bbox_token_idx", 32002)
        self.seg_token_idx = kwargs.pop("seg_token_idx")
        
        # Dual-Stream: Consistency stream configuration
        config.use_consistency_stream = kwargs.pop(
            "use_consistency_stream",
            getattr(config, "use_consistency_stream", False),
        )
        config.vae_path = kwargs.pop(
            "vae_path",
            getattr(config, "vae_path", "stabilityai/sd-vae-ft-mse"),
        )
        config.consistency_encoder_type = kwargs.pop(
            "consistency_encoder_type",
            getattr(config, "consistency_encoder_type", "clip"),
        )


    def _initialize_loss_weights(self, kwargs):
        self.ce_loss_weight = kwargs.pop("ce_loss_weight", 1.0)
        self.dice_loss_weight = kwargs.pop("dice_loss_weight", 0.0)
        self.bce_loss_weight = kwargs.pop("bce_loss_weight", 0.0)
        self.scama_loss_weight = kwargs.pop("scama_loss_weight", 0.0)

    def get_grounding_encoder_embs(self, pixel_values: torch.FloatTensor):
        with torch.no_grad():
            return torch.cat([self._encode_single_image(img) for img in pixel_values], dim=0)

    def _encode_single_image(self, image):
        torch.cuda.empty_cache()
        return self.model.grounding_encoder.image_encoder(image.unsqueeze(0))

    def forward(self, **kwargs):
        # Debug disabled - too verbose
        # import os
        # debug = os.environ.get("LAP_FORENSIC_DEBUG", "0") != "0"
        # if debug and not hasattr(self, '_forward_call_count'):
        #     self._forward_call_count = 0
        # if debug:
        #     self._forward_call_count += 1
        #     if self._forward_call_count <= 5:
        #         ...
        
        # Check if past_key_values has actual content (not just an empty DynamicCache)
        # First generation step: past_key_values is None or empty DynamicCache, should use model_forward to process images
        # Subsequent steps: past_key_values has content, should use super().forward for efficient decoding
        past_key_values = kwargs.get("past_key_values")
        has_past_content = False
        if past_key_values is not None:
            if hasattr(past_key_values, 'get_seq_length'):
                # DynamicCache or similar
                has_past_content = past_key_values.get_seq_length() > 0
            elif isinstance(past_key_values, tuple) and len(past_key_values) > 0:
                # Legacy tuple format
                has_past_content = past_key_values[0] is not None and len(past_key_values[0]) > 0
            else:
                has_past_content = True  # Assume it has content if we can't determine
        
        if has_past_content:
            # Decoding phase: images are already encoded in past_key_values
            # Call super().forward() (LlavaLlamaForCausalLM) which has an early return path
            # in prepare_inputs_labels_for_multimodal for when input_ids.shape[1] == 1
            # IMPORTANT: Keep images in kwargs! The early return path needs images is not None
            # to correctly update attention_mask for the KV cache length
            # Do NOT remove images - early return path needs images to update attention_mask
            if "bboxes" in kwargs:
                _ = kwargs.pop("bboxes")
            return super().forward(**kwargs)
        else:
            # First generation step: need to process images
            return self.model_forward(**kwargs)

    def model_forward(self, global_enc_images: Optional[torch.FloatTensor] = None, 
                      grounding_enc_images: Optional[torch.FloatTensor] = None,
                      bboxes: Optional[Union[torch.FloatTensor, List]] = None, 
                      input_ids: Optional[torch.LongTensor] = None, 
                      labels: Optional[torch.LongTensor] = None,
                      attention_masks: Optional[torch.LongTensor] = None, 
                      offset: Optional[torch.LongTensor] = None, 
                      masks_list: Optional[List[torch.FloatTensor]] = None,
                      label_list: Optional[List[torch.Tensor]] = None, 
                      resize_list: Optional[List[tuple]] = None, 
                      inference: bool = False,
                      use_left_padding: bool = False,  # RL generation flag for left-padded inputs
                      **kwargs, ):
        
        # Handle multiple possible image argument names
        # Priority: global_enc_images > images > pixel_values
        if global_enc_images is None:
            if "images" in kwargs:
                global_enc_images = kwargs.pop("images")
            elif "pixel_values" in kwargs:
                global_enc_images = kwargs.pop("pixel_values")
            
        if input_ids is None:
             # Try to find input_ids in kwargs
             if "input_ids" in kwargs:
                 input_ids = kwargs["input_ids"]
             else:
                 print(f"DEBUG: model_forward missing input_ids. keys: {kwargs.keys()}")
                 return None # Or raise error
        
        # Handle attention_mask (singular) vs attention_masks (plural) naming
        # prepare_inputs_for_generation passes 'attention_mask', but this function expects 'attention_masks'
        if attention_masks is None:
            if "attention_mask" in kwargs:
                attention_masks = kwargs.pop("attention_mask")
        
        # Handle use_left_padding from kwargs (for RL generation)
        if "use_left_padding" in kwargs:
            use_left_padding = kwargs.pop("use_left_padding")

        batch_size = input_ids.shape[0]

        # 1. Handle offset
        if offset is None:
            # Assume 1-to-1 mapping if offset is missing
            offset = torch.arange(batch_size + 1, dtype=torch.long, device=input_ids.device)
        
        # 2. Handle global_enc_images expansion (if batch expanded by GRPOTrainer)
        if global_enc_images is not None and global_enc_images.shape[0] != batch_size:
            src_size = global_enc_images.shape[0]
            if batch_size % src_size == 0:
                 ratio = batch_size // src_size
                 # Repeat interleave to match [prompt1, prompt1, ..., prompt2, prompt2, ...]
                 global_enc_images = global_enc_images.repeat_interleave(ratio, dim=0)
            else:
                 print(f"DEBUG: Shape mismatch: input_ids {batch_size}, images {src_size}")
        
        # Fallback: Create dummy images if missing (to prevent crash)
        if global_enc_images is None:
            # Check if model has vision tower
            vision_tower = getattr(self.model, "vision_tower", None)
            if vision_tower is not None:
                # print("DEBUG: global_enc_images is None! Creating dummy images.")
                # Default CLIP size 336x336
                global_enc_images = torch.zeros((batch_size, 3, 336, 336), device=input_ids.device, dtype=self.parameters().__next__().dtype)

        # 3. Handle masks_list expansion
        if masks_list is not None and len(masks_list) != batch_size and len(masks_list) > 0:
             src_size = len(masks_list)
             if batch_size % src_size == 0:
                 ratio = batch_size // src_size
                 new_masks_list = []
                 for m in masks_list:
                     new_masks_list.extend([m] * ratio)
                 masks_list = new_masks_list
        
        # 4. Handle bboxes expansion
        if bboxes is not None and isinstance(bboxes, list) and len(bboxes) != batch_size and len(bboxes) > 0:
             src_size = len(bboxes)
             if batch_size % src_size == 0:
                 ratio = batch_size // src_size
                 new_bboxes = []
                 for b in bboxes:
                     new_bboxes.extend([b] * ratio)
                 bboxes = new_bboxes
                 
        # 5. Handle resize_list expansion
        if resize_list is not None and len(resize_list) != batch_size and len(resize_list) > 0:
             src_size = len(resize_list)
             if batch_size % src_size == 0:
                 ratio = batch_size // src_size
                 new_resize_list = []
                 for r in resize_list:
                     new_resize_list.extend([r] * ratio)
                 resize_list = new_resize_list
                 
        # 6. Handle label_list expansion
        if label_list is not None and len(label_list) != batch_size and len(label_list) > 0:
             src_size = len(label_list)
             if batch_size % src_size == 0:
                 ratio = batch_size // src_size
                 new_label_list = []
                 for l in label_list:
                     new_label_list.extend([l] * ratio)
                 label_list = new_label_list

        # Handle inference or training paths
        if inference:
            output_hidden_states = self._inference_path(input_ids, global_enc_images, attention_masks)
        else:
            output, output_hidden_states = self._training_path(
                global_enc_images, bboxes, input_ids, labels, attention_masks, offset,
                use_left_padding=use_left_padding,
                use_cache=kwargs.get("use_cache"),
                past_key_values=kwargs.get("past_key_values")  # Pass cache for in-place update
            )
            if labels is None and masks_list is None and grounding_enc_images is None:
                return output
        if grounding_enc_images is not None:
            # Extract grounding encoder image embeddings
            image_embeddings = self.get_grounding_encoder_embs(grounding_enc_images)
            assert image_embeddings.shape[0] == len(offset) - 1

            # Create segmentation token mask (use actual hidden_states length for alignment)
            hidden_states_seq_len = output_hidden_states[-1].shape[1]
            seg_token_mask = self._create_seg_token_mask(input_ids, hidden_states_seq_len)

            # Process hidden states
            hidden_states, pred_embeddings = self._process_hidden_states(output_hidden_states, seg_token_mask, offset)

            # Generate and post-process masks
            pred_masks = self._generate_and_postprocess_masks(
                pred_embeddings, image_embeddings, resize_list, label_list
            )

            if inference:
                return {"pred_masks": pred_masks, "gt_masks": masks_list, }
        else:
            pred_masks = None

        # Calculate losses (pass input_ids for SCAMA loss computation)
        return self._calculate_losses(pred_masks, masks_list, output, input_ids)

    def _create_seg_token_mask(self, input_ids, hidden_states_seq_len):
        """
        Create segmentation token mask aligned with hidden states.
        
        The hidden_states have a different sequence length than input_ids because
        LLaVA replaces the single image placeholder token with 576 actual image tokens.
        This method dynamically computes the padding needed to align the mask.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            hidden_states_seq_len: Actual sequence length of hidden states
        """
        # Create mask for SEG tokens (skip first token which is usually BOS)
        mask = input_ids[:, 1:] == self.seg_token_idx
        
        # Calculate padding needed to match hidden_states length
        # hidden_states_seq_len = original_input_len - 1 (image placeholder) + 576 (image tokens)
        # mask length after input_ids[:, 1:] = original_input_len - 1
        # So we need to pad: hidden_states_seq_len - mask.shape[1]
        total_padding = hidden_states_seq_len - mask.shape[1]
        
        # Padding goes at the front (for image tokens) with 1 token slack at the end
        front_padding = total_padding - 1 if total_padding > 0 else 0
        back_padding = 1 if total_padding > 0 else 0
        
        return torch.cat(
            [torch.zeros((mask.shape[0], front_padding), dtype=torch.bool, device=mask.device), 
             mask, 
             torch.zeros((mask.shape[0], back_padding), dtype=torch.bool, device=mask.device)],
            dim=1
        )

    def _inference_path(self, input_ids, global_enc_images, attention_masks):
        length = input_ids.shape[0]
        global_enc_images_extended = global_enc_images.expand(length, -1, -1, -1).contiguous()

        # Process and return inference output
        output_hidden_states = []
        for i in range(input_ids.shape[0]):
            output_i = super().forward(
                images=global_enc_images_extended[i:i + 1], attention_mask=attention_masks[i:i + 1],
                input_ids=input_ids[i:i + 1], output_hidden_states=True, )
            output_hidden_states.append(output_i.hidden_states)
            torch.cuda.empty_cache()

        output_hidden_states = torch.cat(output_hidden_states, dim=0)
        output_hidden_states = [output_hidden_states]
        return output_hidden_states

    def _training_path(self, global_enc_images, bboxes, input_ids, labels, attention_masks, offset, use_left_padding=False, use_cache=None, past_key_values=None):
        global_enc_images = self._prepare_global_enc_image(global_enc_images, offset)
        bboxes_list = bboxes

        # Request attention outputs if SCAMA loss is enabled
        output_attentions = self.scama_loss_weight > 0
        
        output = super().forward(
            images=global_enc_images, attention_mask=attention_masks, input_ids=input_ids, labels=labels,
            output_hidden_states=True, output_attentions=output_attentions, bboxes=bboxes_list,
            use_left_padding=use_left_padding,  # Critical for RL generation with left-padded inputs
            use_cache=use_cache,
            past_key_values=past_key_values,  # CRITICAL: Pass through for in-place cache update
        )
        output_hidden_states = output.hidden_states
        return output, output_hidden_states

    def _prepare_global_enc_image(self, global_enc_image, offset):
        global_enc_image_list = []
        for i in range(len(offset) - 1):
            start_i, end_i = offset[i], offset[i + 1]
            global_enc_image_i = global_enc_image[i].unsqueeze(0).expand(end_i - start_i, -1, -1, -1).contiguous()
            global_enc_image_list.append(global_enc_image_i)
        return torch.cat(global_enc_image_list, dim=0)

    def _process_hidden_states(self, output_hidden_states, seg_token_mask, offset, infer=False):
        hidden_states = [self.model.text_hidden_fcs[0](output_hidden_states[-1])]
        last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
        pred_embeddings = last_hidden_state[seg_token_mask]
        seg_token_counts = seg_token_mask.int().sum(-1)

        seg_token_offset = seg_token_counts.cumsum(-1)
        seg_token_offset = torch.cat([torch.zeros(1).long().cuda(), seg_token_offset], dim=0)
        if not infer:
            seg_token_offset = seg_token_offset[offset]

        pred_embeddings_list = []
        for i in range(len(seg_token_offset) - 1):
            start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
            pred_embeddings_list.append(pred_embeddings[start_i:end_i])
        return hidden_states, pred_embeddings_list

    def _generate_and_postprocess_masks(self, pred_embeddings, image_embeddings, resize_list, label_list, infer=False):
        pred_masks = []
        for i, pred_embedding in enumerate(pred_embeddings):
            sparse_embeddings, dense_embeddings = self.model.grounding_encoder.prompt_encoder(
                points=None, boxes=None, masks=None, text_embeds=pred_embedding.unsqueeze(1)
            )
            sparse_embeddings = sparse_embeddings.to(pred_embedding.dtype)
            low_res_masks, _ = self.model.grounding_encoder.mask_decoder(
                image_embeddings=image_embeddings[i].unsqueeze(0),
                image_pe=self.model.grounding_encoder.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings, dense_prompt_embeddings=dense_embeddings,
                multimask_output=False, )
            orig_size = label_list[i].shape if not infer else label_list[i]
            # During inference, we have original size list in place of label list
            pred_mask = self.model.grounding_encoder.postprocess_masks(
                low_res_masks, input_size=resize_list[i], original_size=orig_size, )
            pred_masks.append(pred_mask[:, 0])
        return pred_masks

    def _calculate_losses(self, pred_masks, masks_list, output, input_ids=None):
        loss_components = self._compute_loss_components(pred_masks, masks_list, output, input_ids)
        return loss_components

    def _compute_loss_components(self, pred_masks, masks_list, output, input_ids=None):
        # Initialize loss components
        if hasattr(output, "logits") and output.logits is not None:
            device = output.logits.device
        else:
            device = self.parameters().__next__().device
        ce_loss = torch.tensor(0.0, device=device)
        if getattr(output, "loss", None) is not None and self.ce_loss_weight is not None:
            ce_loss = output.loss * float(self.ce_loss_weight)
        mask_bce_loss = torch.tensor(0.0, device=ce_loss.device)
        mask_dice_loss = torch.tensor(0.0, device=ce_loss.device)
        scama_loss_value = torch.tensor(0.0, device=ce_loss.device)
        num_masks = 0

        if pred_masks:
            # Iterate over batch and compute mask-related losses
            for batch_idx, pred_mask in enumerate(pred_masks):
                if pred_mask.numel() > 0:  # Ensure pred_mask is not empty
                    gt_mask = masks_list[batch_idx]
                    
                    # Align mask counts: take minimum of gt and pred to handle mismatches
                    min_masks = min(gt_mask.shape[0], pred_mask.shape[0])
                    if min_masks == 0:
                        continue
                    
                    gt_mask = gt_mask[:min_masks]
                    pred_mask = pred_mask[:min_masks]

                    # Compute Binary Cross-Entropy Loss
                    mask_bce_loss += (compute_sigmoid_cross_entropy(pred_mask, gt_mask, mask_count=gt_mask.shape[0]) * gt_mask.shape[0])
                    # Compute Dice Loss
                    mask_dice_loss += (
                            calculate_dice_loss(pred_mask, gt_mask, mask_count=gt_mask.shape[0]) * gt_mask.shape[0])
                    num_masks += gt_mask.shape[0]

        # Normalize the losses
        mask_bce_loss = float(self.bce_loss_weight) * mask_bce_loss / (num_masks + 1e-8)
        mask_dice_loss = float(self.dice_loss_weight) * mask_dice_loss / (num_masks + 1e-8)
        mask_loss = mask_bce_loss + mask_dice_loss
        
        # Compute SCAMA loss if enabled and attention maps are available
        if (self.scama_loss_weight > 0 and input_ids is not None and 
            hasattr(output, 'attentions') and output.attentions is not None):
            # Compute SCAMA loss using attention maps and GT masks
            scama_loss_value = scama_loss(
                attentions=output.attentions,
                input_ids=input_ids,
                gt_masks=masks_list,
                seg_token_idx=self.seg_token_idx,
                special_token_indices=None,
                image_token_length=576,
                attention_threshold=0.5
            )
            scama_loss_value = self.scama_loss_weight * scama_loss_value

        # Aggregate all loss components
        total_loss = ce_loss + mask_loss + scama_loss_value
        return {"loss": total_loss, "ce_loss": ce_loss, "mask_bce_loss": mask_bce_loss,
                "mask_dice_loss": mask_dice_loss, "mask_loss": mask_loss, "scama_loss": scama_loss_value}

    def evaluate(self, global_enc_images, grounding_enc_images, input_ids, resize_list, orig_sizes, max_tokens_new=32,
                 bboxes=None, ):
        with torch.no_grad():
            generation_outputs = self.generate(
                images=global_enc_images, input_ids=input_ids, bboxes=bboxes, max_new_tokens=max_tokens_new,
                num_beams=1, output_hidden_states=True, return_dict_in_generate=True, )

            output_hidden_states = generation_outputs.hidden_states
            generated_output_ids = generation_outputs.sequences

            seg_token_mask = generated_output_ids[:, 1:] == self.seg_token_idx
            # Get actual hidden_states length and compute dynamic padding
            # output_hidden_states is a tuple of tuples for generation, get length from concatenated states
            if isinstance(output_hidden_states[0], tuple):
                # Generation mode: stack all step outputs
                last_hidden = torch.cat([h[-1] for h in output_hidden_states], dim=1)
            else:
                last_hidden = output_hidden_states[-1]
            hidden_states_seq_len = last_hidden.shape[1]
            total_padding = hidden_states_seq_len - seg_token_mask.shape[1]
            front_padding = total_padding - 1 if total_padding > 0 else 0
            back_padding = 1 if total_padding > 0 else 0
            seg_token_mask = torch.cat(
                [torch.zeros((seg_token_mask.shape[0], front_padding), dtype=torch.bool, device=seg_token_mask.device), 
                 seg_token_mask,
                 torch.zeros((seg_token_mask.shape[0], back_padding), dtype=torch.bool, device=seg_token_mask.device)], 
                dim=1)
            # Process hidden states
            hidden_states, predicted_embeddings = self._process_hidden_states(
                output_hidden_states, seg_token_mask, None, infer=True
            )
            image_embeddings = self.get_grounding_encoder_embs(grounding_enc_images)
            # Generate and post-process masks
            pred_masks = self._generate_and_postprocess_masks(
                predicted_embeddings, image_embeddings, resize_list, orig_sizes, infer=True
            )
        return generated_output_ids, pred_masks
    
class LaPForClassification(LlavaLlamaForCausalLM):
    def __init__(self, config, **kwargs):
        self._set_model_configurations(config, kwargs)
        super().__init__(config)
        self.model = LaPModel(config, **kwargs)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.prediction_head = nn.Sequential(
            nn.Linear(config.mm_hidden_size, 2 * config.mm_hidden_size),  
            nn.ReLU(),                                                      
            nn.Linear(2 * config.mm_hidden_size, 2)                        
        )
        self.post_init()

    def _set_model_configurations(self, config, kwargs):
        config.mm_use_image_start_end = kwargs.pop("use_mm_start_end", True)
        config.mm_vision_module = kwargs.get("vision_module", "openai/clip-vit-large-patch14-336")
        self._initialize_loss_weights(kwargs)
        config.bbox_token_idx = kwargs.get("bbox_token_idx", 1)
        config.num_reg_features = kwargs.get("num_level_reg_features", 4)
        config.with_region = kwargs.get("with_region", True)
        config.bbox_token_idx = kwargs.get("bbox_token_idx", 32002)
        self.seg_token_idx = kwargs.pop("seg_token_idx")

    def _initialize_loss_weights(self, kwargs):
        self.ce_loss_weight = kwargs.pop("ce_loss_weight", None)
        self.dice_loss_weight = kwargs.pop("dice_loss_weight", None)
        self.bce_loss_weight = kwargs.pop("bce_loss_weight", None)

    def get_grounding_encoder_embs(self, pixel_values: torch.FloatTensor):
        with torch.no_grad():
            return torch.cat([self._encode_single_image(img) for img in pixel_values], dim=0)

    def _encode_single_image(self, image):
        torch.cuda.empty_cache()
        return self.model.grounding_encoder.image_encoder(image.unsqueeze(0))

    def forward(self, **kwargs):
        train_cls = kwargs.get('train_cls', False)
        if train_cls == True:
            return self.forward_train_cls(**kwargs)
        elif kwargs.get('inference_cls', False) == True:
            return self.forward_inference_cls(**kwargs)
        else:
            return super().forward(**kwargs) if "past_key_values" in kwargs else self.model_forward(**kwargs)
        
    def _calculate_cls_losses(self, outputs, labels):
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(outputs, labels)
        return loss

    def forward_train_cls(self, global_enc_images: torch.FloatTensor, cls_gt_list: List[torch.Tensor], **kwargs, ):
        image_features = self.get_model().get_vision_tower().vision_tower(global_enc_images.to(device=self.device,dtype=self.dtype),output_hidden_states=True,)
        image_feature = image_features.hidden_states[-2]
        cls_feature = image_feature[:, 0, :] 
        logits = self.prediction_head(cls_feature)
        _, cls = torch.max(logits, dim=1)

        # pdb.set_trace()
        return {'loss': self._calculate_cls_losses(logits, cls_gt_list), 'logits': logits}
    
    def forward_inference_cls(self, global_enc_images: torch.FloatTensor , **kwargs, ):
        image_features = self.get_model().get_vision_tower().vision_tower(global_enc_images.to(device=self.device,dtype=self.dtype),output_hidden_states=True,)
        image_feature = image_features.hidden_states[-2]
        cls_feature = image_feature[:, 0, :] 
        logits = self.prediction_head(cls_feature)

        # pdb.set_trace()
        return {'logits': logits}
    
    def model_forward(self, global_enc_images: torch.FloatTensor, grounding_enc_images: torch.FloatTensor,
                      bboxes: torch.FloatTensor, input_ids: torch.LongTensor, labels: torch.LongTensor,
                      attention_masks: torch.LongTensor, offset: torch.LongTensor, masks_list: List[torch.FloatTensor],
                      label_list: List[torch.Tensor], resize_list: List[tuple], inference: bool = False, **kwargs, ):

        # Handle inference or training paths
        if inference:
            output_hidden_states, cls_output = self._inference_path(input_ids, global_enc_images, attention_masks)
        else:
            output, output_hidden_states, cls_output = self._training_path(
                global_enc_images, bboxes, input_ids, labels, attention_masks, offset
            )
        if grounding_enc_images is not None:
            # Extract grounding encoder image embeddings
            image_embeddings = self.get_grounding_encoder_embs(grounding_enc_images)
            assert image_embeddings.shape[0] == len(offset) - 1

            # Create segmentation token mask (use actual hidden_states length for alignment)
            hidden_states_seq_len = output_hidden_states[-1].shape[1]
            seg_token_mask = self._create_seg_token_mask(input_ids, hidden_states_seq_len)

            # Process hidden states
            hidden_states, pred_embeddings = self._process_hidden_states(output_hidden_states, seg_token_mask, offset)

            # Generate and post-process masks
            pred_masks = self._generate_and_postprocess_masks(
                pred_embeddings, image_embeddings, resize_list, label_list
            )

            if inference:
                return {"pred_masks": pred_masks, "gt_masks": masks_list, "cls": cls_output[0]}
        else:
            pred_masks = None

        # Calculate losses
        return self._calculate_losses(pred_masks, masks_list, output)

    def _create_seg_token_mask(self, input_ids, hidden_states_seq_len):
        """
        Create segmentation token mask aligned with hidden states.
        
        The hidden_states have a different sequence length than input_ids because
        LLaVA replaces the single image placeholder token with 576 actual image tokens.
        """
        mask = input_ids[:, 1:] == self.seg_token_idx
        
        # Calculate padding needed to match hidden_states length
        total_padding = hidden_states_seq_len - mask.shape[1]
        front_padding = total_padding - 1 if total_padding > 0 else 0
        back_padding = 1 if total_padding > 0 else 0
        
        return torch.cat(
            [torch.zeros((mask.shape[0], front_padding), dtype=torch.bool, device=mask.device), 
             mask, 
             torch.zeros((mask.shape[0], back_padding), dtype=torch.bool, device=mask.device)],
            dim=1
        )

    def _inference_path(self, input_ids, global_enc_images, attention_masks):
        length = input_ids.shape[0]
        global_enc_images_extended = global_enc_images.expand(length, -1, -1, -1).contiguous()

        # cls real or fake
        image_features = self.get_model().get_vision_tower().vision_tower(global_enc_images.to(device=self.device,dtype=self.dtype),output_hidden_states=True,)
        image_feature = image_features.hidden_states[-2]
        cls_feature = image_feature[:, 0, :] 
        logit = self.prediction_head(cls_feature)
        _, cls = torch.max(logit, dim=1)

        # Process and return inference output
        output_hidden_states = []
        for i in range(input_ids.shape[0]):
            output_i = super().forward(
                images=global_enc_images_extended[i:i + 1], attention_mask=attention_masks[i:i + 1],
                input_ids=input_ids[i:i + 1], output_hidden_states=True, )
            output_hidden_states.append(output_i.hidden_states)
            torch.cuda.empty_cache()

        output_hidden_states = torch.cat(output_hidden_states, dim=0)
        output_hidden_states = [output_hidden_states]
        return output_hidden_states, (cls, logit)

    def _training_path(self, global_enc_images, bboxes, input_ids, labels, attention_masks, offset):
        global_enc_images = self._prepare_global_enc_image(global_enc_images, offset)
        bboxes_list = bboxes
        # cls real or fake
        image_features = self.get_model().get_vision_tower().vision_tower(global_enc_images.to(device=self.device,dtype=self.dtype),output_hidden_states=True,)
        image_feature = image_features.hidden_states[-2]
        cls_feature = image_feature[:, 0, :] 
        logit = self.prediction_head(cls_feature)
        _, cls = torch.max(logit, dim=1)
        # pdb.set_trace()
        output = super().forward(
            images=global_enc_images, attention_mask=attention_masks, input_ids=input_ids, labels=labels,
            output_hidden_states=True, bboxes=bboxes_list, )
        output_hidden_states = output.hidden_states
        return output, output_hidden_states, (cls, logit)

    def _prepare_global_enc_image(self, global_enc_image, offset):
        global_enc_image_list = []
        for i in range(len(offset) - 1):
            start_i, end_i = offset[i], offset[i + 1]
            global_enc_image_i = global_enc_image[i].unsqueeze(0).expand(end_i - start_i, -1, -1, -1).contiguous()
            global_enc_image_list.append(global_enc_image_i)
        return torch.cat(global_enc_image_list, dim=0)

    def _process_hidden_states(self, output_hidden_states, seg_token_mask, offset, infer=False):
        hidden_states = [self.model.text_hidden_fcs[0](output_hidden_states[-1])]
        last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
        pred_embeddings = last_hidden_state[seg_token_mask]
        seg_token_counts = seg_token_mask.int().sum(-1)

        seg_token_offset = seg_token_counts.cumsum(-1)
        seg_token_offset = torch.cat([torch.zeros(1).long().cuda(), seg_token_offset], dim=0)
        if not infer:
            seg_token_offset = seg_token_offset[offset]

        pred_embeddings_list = []
        for i in range(len(seg_token_offset) - 1):
            start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
            pred_embeddings_list.append(pred_embeddings[start_i:end_i])
        return hidden_states, pred_embeddings_list

    def _generate_and_postprocess_masks(self, pred_embeddings, image_embeddings, resize_list, label_list, infer=False):
        pred_masks = []
        for i, pred_embedding in enumerate(pred_embeddings):
            sparse_embeddings, dense_embeddings = self.model.grounding_encoder.prompt_encoder(
                points=None, boxes=None, masks=None, text_embeds=pred_embedding.unsqueeze(1)
            )
            sparse_embeddings = sparse_embeddings.to(pred_embedding.dtype)
            low_res_masks, _ = self.model.grounding_encoder.mask_decoder(
                image_embeddings=image_embeddings[i].unsqueeze(0),
                image_pe=self.model.grounding_encoder.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings, dense_prompt_embeddings=dense_embeddings,
                multimask_output=False, )
            orig_size = label_list[i].shape if not infer else label_list[i]
            # During inference, we have original size list in place of label list
            pred_mask = self.model.grounding_encoder.postprocess_masks(
                low_res_masks, input_size=resize_list[i], original_size=orig_size, )
            pred_masks.append(pred_mask[:, 0])
        return pred_masks

    def _calculate_losses(self, pred_masks, masks_list, output):
        loss_components = self._compute_loss_components(pred_masks, masks_list, output)
        return loss_components

    def _compute_loss_components(self, pred_masks, masks_list, output):
        # Initialize loss components
        ce_loss = output.loss * self.ce_loss_weight
        mask_bce_loss = torch.tensor(0.0, device=ce_loss.device)
        mask_dice_loss = torch.tensor(0.0, device=ce_loss.device)
        num_masks = 0

        if pred_masks:
            # Iterate over batch and compute mask-related losses
            for batch_idx, pred_mask in enumerate(pred_masks):
                if pred_mask.numel() > 0:  # Ensure pred_mask is not empty
                    gt_mask = masks_list[batch_idx]
                    
                    # Align mask counts: take minimum of gt and pred to handle mismatches
                    min_masks = min(gt_mask.shape[0], pred_mask.shape[0])
                    if min_masks == 0:
                        continue
                    
                    gt_mask = gt_mask[:min_masks]
                    pred_mask = pred_mask[:min_masks]

                    # Compute Binary Cross-Entropy Loss
                    mask_bce_loss += (compute_sigmoid_cross_entropy(pred_mask, gt_mask, mask_count=gt_mask.shape[0]) *
                                      gt_mask.shape[0])
                    # Compute Dice Loss
                    mask_dice_loss += (
                            calculate_dice_loss(pred_mask, gt_mask, mask_count=gt_mask.shape[0]) * gt_mask.shape[0])
                    num_masks += gt_mask.shape[0]

        # Normalize the losses
        mask_bce_loss = self.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
        mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)
        mask_loss = mask_bce_loss + mask_dice_loss

        # Aggregate all loss components
        total_loss = ce_loss + mask_loss
        return {"loss": total_loss, "ce_loss": ce_loss, "mask_bce_loss": mask_bce_loss,
                "mask_dice_loss": mask_dice_loss, "mask_loss": mask_loss, }

    def evaluate(self, global_enc_images, grounding_enc_images, input_ids, resize_list, orig_sizes, max_tokens_new=32,
                 bboxes=None, ):
        with torch.no_grad():
            generation_outputs = self.generate(
                images=global_enc_images, input_ids=input_ids, bboxes=bboxes, max_new_tokens=max_tokens_new,
                num_beams=1, output_hidden_states=True, return_dict_in_generate=True, )

            output_hidden_states = generation_outputs.hidden_states
            generated_output_ids = generation_outputs.sequences

            seg_token_mask = generated_output_ids[:, 1:] == self.seg_token_idx
            # Get actual hidden_states length and compute dynamic padding
            if isinstance(output_hidden_states[0], tuple):
                last_hidden = torch.cat([h[-1] for h in output_hidden_states], dim=1)
            else:
                last_hidden = output_hidden_states[-1]
            hidden_states_seq_len = last_hidden.shape[1]
            total_padding = hidden_states_seq_len - seg_token_mask.shape[1]
            front_padding = total_padding - 1 if total_padding > 0 else 0
            back_padding = 1 if total_padding > 0 else 0
            seg_token_mask = torch.cat(
                [torch.zeros((seg_token_mask.shape[0], front_padding), dtype=torch.bool, device=seg_token_mask.device), 
                 seg_token_mask,
                 torch.zeros((seg_token_mask.shape[0], back_padding), dtype=torch.bool, device=seg_token_mask.device)], 
                dim=1)
            # Process hidden states
            hidden_states, predicted_embeddings = self._process_hidden_states(
                output_hidden_states, seg_token_mask, None, infer=True
            )
            image_embeddings = self.get_grounding_encoder_embs(grounding_enc_images)
            # Generate and post-process masks
            pred_masks = self._generate_and_postprocess_masks(
                predicted_embeddings, image_embeddings, resize_list, orig_sizes, infer=True
            )
        return generated_output_ids, pred_masks
