import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, Tuple

from tools.utils import IGNORE_INDEX, IMAGE_TOKEN_INDEX
from model.layers import MLVLROIQueryModule
from model.llava.model.multimodal_encoder.builder import build_vision_tower


class LlavaMetaModel:
    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            modules = [nn.Linear(config.mm_hidden_size, config.hidden_size),
                       nn.GELU(),
                       nn.Linear(config.hidden_size, config.hidden_size)]
            self.mm_projector = nn.Sequential(*modules)
        self.region_encoder = MLVLROIQueryModule(embed_dims=1024, out_dims=4096, num_levels=4)
        
        # Dual-Stream: Initialize consistency stream components
        self._init_consistency_stream(config)

    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        self.config.mm_vision_tower = vision_tower

        vision_tower = build_vision_tower(model_args)

        if fsdp is not None and len(fsdp) > 0:
            self.vision_tower = [vision_tower]
        else:
            self.vision_tower = vision_tower

        self.config.use_mm_proj = True
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature

        if not hasattr(self, "mm_projector"):
            self.mm_projector = nn.Linear(
                self.config.mm_hidden_size, self.config.hidden_size
            )

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(
                pretrain_mm_mlp_adapter, map_location="cpu"
            )

            def get_w(weights, keyword):
                return {
                    k.split(keyword + ".")[1]: v
                    for k, v in weights.items()
                    if keyword in k
                }

            self.mm_projector.load_state_dict(
                get_w(mm_projector_weights, "mm_projector")
            )
    
    def _init_consistency_stream(self, config):
        """Initialize dual-stream consistency encoder components."""
        self.use_consistency_stream = getattr(config, "use_consistency_stream", False)
        
        if self.use_consistency_stream:
            from model.consistency_encoder import DiffusionConsistencyModule, ConsistencyFeatureEncoder
            
            vae_path = getattr(config, "vae_path", "stabilityai/sd-vae-ft-mse")
            encoder_type = getattr(config, "consistency_encoder_type", "clip")
            hidden_dim = getattr(config, "mm_hidden_size", 1024)
            
            # Frozen VAE for reconstruction
            self.consistency_module = DiffusionConsistencyModule(vae_path=vae_path)
            
            # Feature encoder for residual maps
            self.consistency_encoder = ConsistencyFeatureEncoder(
                encoder_type=encoder_type,
                hidden_dim=hidden_dim,
                use_shared_clip=(encoder_type == "clip")
            )
            
            # Projection layer for consistency features (matches mm_projector structure)
            self.consistency_projector = nn.Sequential(
                nn.Linear(config.mm_hidden_size, config.hidden_size),
                nn.GELU(),
                nn.Linear(config.hidden_size, config.hidden_size)
            )
    
    def get_consistency_features(
        self, 
        images: torch.Tensor, 
        vision_tower_output: Optional[Tuple] = None
    ) -> Optional[torch.Tensor]:
        """Extract consistency stream features from images."""
        if not self.use_consistency_stream:
            return None
        
        with torch.no_grad():
            _, residual_map = self.consistency_module(images, return_reconstruction=False)
        
        # Use shared CLIP encoder if configured
        encoder_type = getattr(self.config, "consistency_encoder_type", "clip")
        if encoder_type == "clip" and vision_tower_output is not None:
            # Re-encode residual map through CLIP
            residual_features, _ = self.vision_tower(residual_map)
            consistency_features = self.consistency_encoder(residual_map, residual_features)
        else:
            consistency_features = self.consistency_encoder(residual_map)
        
        # Project to LLM hidden dimension
        # Ensure dtype matches consistency_projector weights to avoid float vs bfloat16 mismatch
        target_dtype = next(self.consistency_projector.parameters()).dtype
        consistency_features = consistency_features.to(dtype=target_dtype)
        consistency_features = self.consistency_projector(consistency_features)
        
        return consistency_features


class LlavaMetaForCausalLM(ABC):
    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images):
        image_features, image_forward_outs = self.get_model().get_vision_tower()(images)
        # Ensure image_features dtype matches mm_projector weights to avoid float vs bfloat16 mismatch
        mm_projector = self.get_model().mm_projector
        target_dtype = next(mm_projector.parameters()).dtype
        image_features = image_features.to(dtype=target_dtype)
        image_features = mm_projector(image_features)
        
        # Dual-Stream: Get consistency features if enabled
        consistency_features = None
        if getattr(self.get_model(), 'use_consistency_stream', False):
            consistency_features = self.get_model().get_consistency_features(
                images, vision_tower_output=image_forward_outs
            )
        
        return image_features, image_forward_outs, consistency_features


    def prepare_inputs_labels_for_multimodal(
        self, input_ids, attention_mask, past_key_values, labels, images, bboxes,
        use_left_padding: bool = False
    ):
        """
        Prepare inputs and labels for multimodal forward pass.
        
        Args:
            use_left_padding: If True, handle left-padded batches correctly for generation.
                             SFT uses right-padding, RL generation uses left-padding.
        """
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if (
                past_key_values is not None
                and vision_tower is not None
                and images is not None
                and input_ids.shape[1] == 1
            ):
                attention_mask = torch.ones(
                    (attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
            return input_ids, attention_mask, past_key_values, None, labels

        if type(images) is list or images.ndim == 5:
            concat_images = torch.cat([image for image in images], dim=0)
            image_features, _, consistency_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            image_features = [x.flatten(0, 1) for x in image_features]
            # Handle consistency features split
            if consistency_features is not None:
                consistency_features = torch.split(consistency_features, split_sizes, dim=0)
                consistency_features = [x.flatten(0, 1) for x in consistency_features]
        else:
            # Process for region
            image_features, image_forward_outs, consistency_features = self.encode_images(images)
            if self.config.with_region:
                select_hidden_state_layer = self.config.mm_vision_select_layer
                num_level_reg_features = self.config.num_level_reg_features
                mlvl_reg_features = image_forward_outs.hidden_states[select_hidden_state_layer::-3]
                mlvl_reg_features = mlvl_reg_features[::-1]
                mlvl_reg_features = mlvl_reg_features[-num_level_reg_features:]
                mlvl_reg_features = [item[:, 1:].to(images.dtype) for item in mlvl_reg_features]

                if bboxes is not None and (len(bboxes) > 0):
                    mlvl_reg_features = self.model.region_encoder(mlvl_reg_features, bboxes)
                else:
                    mlvl_reg_features = [None for _ in range(len(input_ids))]


        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_image_idx = 0
        for batch_idx, (cur_input_ids, reg_feat) in enumerate(zip(input_ids, mlvl_reg_features)): # Adjusted the loop to include reg_feat
            curr_full_input_ids = []
            if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:
                # multimodal LLM, but the current sample is not multimodal
                cur_input_embeds = self.get_model().embed_tokens(cur_input_ids)
                # Cast dummy_feature to mm_projector dtype to avoid float vs bfloat16 mismatch
                mm_projector = self.get_model().mm_projector
                dummy_feat = vision_tower.dummy_feature.to(dtype=next(mm_projector.parameters()).dtype)
                cur_input_embeds = (
                    cur_input_embeds
                    + (
                        0.0 * mm_projector(dummy_feat)
                    ).sum()
                )
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue
            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            cur_new_input_embeds = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape
            while image_token_indices.numel() > 0:
                cur_image_features = image_features[cur_image_idx]
                # Dual-Stream: Get consistency features for current image
                cur_consistency_features = None
                if consistency_features is not None:
                    cur_consistency_features = consistency_features[cur_image_idx]
                image_token_start = image_token_indices[0]

                if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(
                    self.config, "mm_use_im_start_end", False
                ):
                    # preparing input embedding
                    cur_new_input_embeds.append(
                        self.get_model()
                        .embed_tokens(cur_input_ids[: image_token_start - 1])
                        .detach()
                    )
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(
                            cur_input_ids[image_token_start - 1 : image_token_start]
                        )
                    )
                    cur_new_input_embeds.append(cur_image_features)
                    # Dual-Stream: Append consistency features after image features
                    if cur_consistency_features is not None:
                        cur_new_input_embeds.append(cur_consistency_features)
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(
                            cur_input_ids[image_token_start + 1 : image_token_start + 2]
                        )
                    )
                    # preparing input_ids
                    curr_full_input_ids.append(cur_input_ids[: image_token_start - 1])
                    curr_full_input_ids.append(cur_input_ids[image_token_start - 1: image_token_start])
                    curr_full_image_token = torch.full((cur_image_features.shape[0],), image_token_start, dtype=torch.int64)
                    curr_full_input_ids.append(curr_full_image_token)
                    # Dual-Stream: Add placeholder IDs for consistency tokens
                    if cur_consistency_features is not None:
                        curr_full_consistency_token = torch.full((cur_consistency_features.shape[0],), image_token_start, dtype=torch.int64)
                        curr_full_input_ids.append(curr_full_consistency_token)
                    curr_full_input_ids.append(cur_input_ids[image_token_start + 1: image_token_start + 2])
                    # preparing labels
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(
                            torch.full(
                                (cur_image_features.shape[0],),
                                IGNORE_INDEX,
                                device=labels.device,
                                dtype=labels.dtype,
                            )
                        )
                        # Dual-Stream: Add IGNORE_INDEX labels for consistency tokens
                        if cur_consistency_features is not None:
                            cur_new_labels.append(
                                torch.full(
                                    (cur_consistency_features.shape[0],),
                                    IGNORE_INDEX,
                                    device=labels.device,
                                    dtype=labels.dtype,
                                )
                            )
                        cur_new_labels.append(
                            cur_labels[image_token_start : image_token_start + 1]
                        )
                        cur_labels = cur_labels[image_token_start + 2 :]
                elif getattr(self.config, "mm_use_im_start_end", False):
                    # preparing input embedding
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_ids[:image_token_start])
                    )
                    cur_new_input_embeds.append(cur_image_features)
                    # Dual-Stream: Append consistency features after image features
                    if cur_consistency_features is not None:
                        cur_new_input_embeds.append(cur_consistency_features)
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(
                            cur_input_ids[image_token_start + 1 : image_token_start + 2]
                        )
                    )
                    # preparing input_ids
                    curr_full_input_ids.append(cur_input_ids[: image_token_start])
                    curr_full_image_token = torch.full((cur_image_features.shape[0],), image_token_start,
                                                       dtype=torch.int64)
                    curr_full_input_ids.append(curr_full_image_token)
                    # Dual-Stream: Add placeholder IDs for consistency tokens
                    if cur_consistency_features is not None:
                        curr_full_consistency_token = torch.full((cur_consistency_features.shape[0],), image_token_start, dtype=torch.int64)
                        curr_full_input_ids.append(curr_full_consistency_token)
                    curr_full_input_ids.append(cur_input_ids[image_token_start + 1: image_token_start + 2])
                    # preparing labels
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(
                            torch.full(
                                (cur_image_features.shape[0],),
                                IGNORE_INDEX,
                                device=labels.device,
                                dtype=labels.dtype,
                            )
                        )
                        # Dual-Stream: Add IGNORE_INDEX labels for consistency tokens
                        if cur_consistency_features is not None:
                            cur_new_labels.append(
                                torch.full(
                                    (cur_consistency_features.shape[0],),
                                    IGNORE_INDEX,
                                    device=labels.device,
                                    dtype=labels.dtype,
                                )
                            )
                        cur_new_labels.append(
                            cur_labels[image_token_start + 1 : image_token_start + 2]
                        )
                        cur_labels = cur_labels[image_token_start + 2 :]
                else:
                    # preparing input embedding
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_ids[:image_token_start])
                    )
                    cur_new_input_embeds.append(cur_image_features)
                    # Dual-Stream: Append consistency features after image features
                    if cur_consistency_features is not None:
                        cur_new_input_embeds.append(cur_consistency_features)
                    # preparing input_ids
                    curr_full_input_ids.append(cur_input_ids[: image_token_start])
                    curr_full_image_token = torch.full((cur_image_features.shape[0],), image_token_start,
                                                       dtype=torch.int64)
                    curr_full_input_ids.append(curr_full_image_token)
                    # Dual-Stream: Add placeholder IDs for consistency tokens
                    if cur_consistency_features is not None:
                        curr_full_consistency_token = torch.full((cur_consistency_features.shape[0],), image_token_start, dtype=torch.int64)
                        curr_full_input_ids.append(curr_full_consistency_token)
                    # preparing labels
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(
                            torch.full(
                                (cur_image_features.shape[0],),
                                IGNORE_INDEX,
                                device=labels.device,
                                dtype=labels.dtype,
                            )
                        )
                        # Dual-Stream: Add IGNORE_INDEX labels for consistency tokens
                        if cur_consistency_features is not None:
                            cur_new_labels.append(
                                torch.full(
                                    (cur_consistency_features.shape[0],),
                                    IGNORE_INDEX,
                                    device=labels.device,
                                    dtype=labels.dtype,
                                )
                            )
                        cur_labels = cur_labels[image_token_start + 1 :]

                cur_image_idx += 1
                if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(
                    self.config, "mm_use_im_start_end", False
                ):
                    cur_input_ids = cur_input_ids[image_token_start + 2 :]
                elif getattr(self.config, "mm_use_im_start_end", False):
                    cur_input_ids = cur_input_ids[image_token_start + 2 :]
                else:
                    cur_input_ids = cur_input_ids[image_token_start + 1 :]
                image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            if cur_input_ids.numel() > 0:
                if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(
                    self.config, "mm_use_im_start_end", False
                ):
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_ids).detach()
                    )
                elif getattr(self.config, "mm_use_im_start_end", False):
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_ids)
                    )
                else:
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_ids)
                    )
                curr_full_input_ids.append(cur_input_ids)
                if labels is not None:
                    cur_new_labels.append(cur_labels)
            cur_new_input_embeds = [
                x.to(device=self.device) for x in cur_new_input_embeds
            ]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            curr_full_input_ids = [x.to(device=self.device) for x in curr_full_input_ids]
            curr_full_input_ids = torch.cat(curr_full_input_ids, dim=0)
            # current new_input_embeds computation complete (Lx4096)
            # Replace embeds of <bbox> with region feats (num_box x 4096)
            if reg_feat is not None:
                BBOX_TOKEN_ID = self.config.bbox_token_idx
                reg_embeds = torch.zeros_like(cur_new_input_embeds)  # (Lx4096)
                reg_mask = (curr_full_input_ids == BBOX_TOKEN_ID)

                # To Handle errors: Check if the shapes of reg_embeds[reg_mask] and reg_feat match
                if reg_embeds[reg_mask].shape[0] != reg_feat.shape[0]:
                    # If they don't match, slice reg_feat to make the shapes match
                    min_shape = reg_embeds[reg_mask].shape[0]
                    reg_feat = reg_feat[:min_shape]

                reg_embeds[reg_mask] = reg_feat.to(reg_embeds.dtype)
                cur_new_input_embeds = cur_new_input_embeds * (~reg_mask).to(
                    cur_new_input_embeds.dtype)[:, None] + reg_embeds

            new_input_embeds.append(cur_new_input_embeds)
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)

        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat(
                    (
                        cur_new_embed,
                        torch.zeros(
                            (max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]),
                            dtype=cur_new_embed.dtype,
                            device=cur_new_embed.device,
                        ),
                    ),
                    dim=0,
                )
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat(
                        (
                            cur_new_label,
                            torch.full(
                                (max_len - cur_new_label.shape[0],),
                                IGNORE_INDEX,
                                dtype=cur_new_label.dtype,
                                device=cur_new_label.device,
                            ),
                        ),
                        dim=0,
                    )
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)

            if attention_mask is not None:
                new_attention_mask = []
                # Number of new tokens added per sample due to image/region expansion
                # cur_new_labels.shape[0] = length after image expansion (before alignment padding)
                # labels.shape[1] = original input_ids length
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(
                    attention_mask, _new_labels, new_labels
                ):
                    # Tokens added by image/region expansion (should be attended to)
                    num_image_tokens_added = cur_new_labels.shape[0] - labels.shape[1]
                    # Tokens added for alignment padding (should NOT be attended to)
                    num_align_pad = cur_new_labels_align.shape[0] - cur_new_labels.shape[0]
                    
                    if use_left_padding:
                        # For left-padded inputs (RL generation):
                        # Original structure: [PAD PAD PAD ... | CONTENT CONTENT ...]
                        # After expansion:    [PAD PAD PAD ... | IMG_TOKENS ... CONTENT ...]
                        # After alignment:    [PAD PAD PAD ... | IMG_TOKENS ... CONTENT ... | ALIGN_PAD]
                        # 
                        # Key insight: image tokens are inserted within content region,
                        # so we need to preserve leading padding as False
                        
                        # Find where actual content starts (first True/1 in original mask)
                        first_true_idx = (cur_attention_mask == 1).nonzero()
                        if len(first_true_idx) > 0:
                            leading_pad = first_true_idx[0].item()
                        else:
                            leading_pad = 0
                        
                        # Build new attention mask:
                        # [False] * leading_pad (preserve original left padding)
                        # + [True] * num_image_tokens_added (new image/region tokens)
                        # + original content part (cur_attention_mask[leading_pad:])
                        # + [False] * num_align_pad (alignment padding at end)
                        cur_new_attention_mask = torch.cat([
                            torch.zeros(leading_pad, dtype=cur_attention_mask.dtype, device=cur_attention_mask.device),
                            torch.ones(num_image_tokens_added, dtype=cur_attention_mask.dtype, device=cur_attention_mask.device),
                            cur_attention_mask[leading_pad:],
                            torch.zeros(num_align_pad, dtype=cur_attention_mask.dtype, device=cur_attention_mask.device),
                        ])
                    else:
                        # For right-padded inputs (SFT training):
                        # Original structure: [CONTENT CONTENT ... | PAD PAD PAD ...]
                        # After expansion:    [IMG_TOKENS ... CONTENT ... | PAD PAD PAD ...]
                        # After alignment:    [IMG_TOKENS ... CONTENT ... | PAD PAD PAD ... | ALIGN_PAD]
                        new_attn_mask_pad_left = torch.full(
                            (num_image_tokens_added,),
                            True,
                            dtype=attention_mask.dtype,
                            device=attention_mask.device,
                        )
                        new_attn_mask_pad_right = torch.full(
                            (num_align_pad,),
                            False,
                            dtype=attention_mask.dtype,
                            device=attention_mask.device,
                        )
                        cur_new_attention_mask = torch.cat(
                            (
                                new_attn_mask_pad_left,
                                cur_attention_mask,
                                new_attn_mask_pad_right,
                            ),
                            dim=0,
                        )
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                
                # Validation assertions
                assert attention_mask.shape == new_labels.shape, \
                    f"attention_mask shape {attention_mask.shape} != new_labels shape {new_labels.shape}"
                assert attention_mask.shape == new_input_embeds.shape[:2], \
                    f"attention_mask shape {attention_mask.shape} != embeds shape {new_input_embeds.shape[:2]}"
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels = torch.stack(new_labels, dim=0)

            if attention_mask is not None:
                # Calculate padding needed for image token expansion
                pad_size = new_input_embeds.shape[1] - input_ids.shape[1]
                
                if use_left_padding:
                    # For left-padded inputs (RL generation):
                    # The image tokens are inserted within the content, but we need to
                    # preserve the original padding structure. The new image tokens
                    # should be attended to (True), but we need to handle per-sample.
                    new_attention_mask = []
                    for i in range(attention_mask.shape[0]):
                        cur_attn = attention_mask[i]  # Original attention mask
                        # Count how many leading zeros (left padding) in original
                        # These should remain as False in the new attention mask
                        # The image tokens are inserted after the first real content
                        # So we add pad_size True values after the original False values
                        
                        # Find where actual content starts (first True)
                        first_true_idx = (cur_attn == 1).nonzero()
                        if len(first_true_idx) > 0:
                            leading_pad = first_true_idx[0].item()
                        else:
                            leading_pad = 0
                        
                        # Build new attention mask:
                        # [False] * leading_pad + [True] * pad_size + original content part
                        new_attn = torch.cat([
                            torch.zeros(leading_pad, dtype=cur_attn.dtype, device=cur_attn.device),
                            torch.ones(pad_size, dtype=cur_attn.dtype, device=cur_attn.device),
                            cur_attn[leading_pad:]
                        ])
                        new_attention_mask.append(new_attn)
                    attention_mask = torch.stack(new_attention_mask)
                else:
                    # Original right-padding logic (SFT training)
                    new_attn_mask_pad_left = torch.full(
                        (
                            attention_mask.shape[0],
                            pad_size,
                        ),
                        True,
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )
                    attention_mask = torch.cat(
                        (new_attn_mask_pad_left, attention_mask), dim=1
                    )
                assert attention_mask.shape == new_input_embeds.shape[:2]

        # DEBUG: Validate attention mask correctness (enable with DEBUG_MULTIMODAL_MASK=1)
        import os
        if os.environ.get("DEBUG_MULTIMODAL_MASK", "0") == "1" and attention_mask is not None:
            batch_size = attention_mask.shape[0]
            seq_len = attention_mask.shape[1]
            print(f"\n[DEBUG prepare_inputs_labels_for_multimodal]", flush=True)
            print(f"  use_left_padding={use_left_padding}", flush=True)
            print(f"  batch_size={batch_size}, seq_len={seq_len}", flush=True)
            print(f"  new_input_embeds.shape={new_input_embeds.shape}", flush=True)
            for i in range(min(batch_size, 2)):  # Print first 2 samples
                mask = attention_mask[i]
                num_zeros = (mask == 0).sum().item()
                num_ones = (mask == 1).sum().item()
                # Find first True and last True positions
                true_positions = (mask == 1).nonzero()
                if len(true_positions) > 0:
                    first_true = true_positions[0].item()
                    last_true = true_positions[-1].item()
                else:
                    first_true, last_true = -1, -1
                print(f"  Sample {i}: mask zeros={num_zeros}, ones={num_ones}, first_true_idx={first_true}, last_true_idx={last_true}", flush=True)
                # Verify padding is at correct position
                if use_left_padding and first_true > 0:
                    leading_mask = mask[:first_true]
                    if leading_mask.sum() > 0:
                        print(f"  ⚠️ WARNING: Leading padding has non-zero mask values!", flush=True)
                    else:
                        print(f"  ✓ Leading padding correctly masked as 0", flush=True)

        return None, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, num_new_tokens):

        if model_args.mm_use_im_start_end:

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(
                    model_args.pretrain_mm_mlp_adapter, map_location="cpu"
                )
                embed_tokens_weight = mm_projector_weights["model.embed_tokens.weight"]
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[
                        -num_new_tokens:
                    ]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(
                        f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}."
                    )
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
