"""
Custom GRPOTrainer for LaP-Forensic multimodal model (V2).

Extends TRL's GRPOTrainer to properly handle image inputs during generation.

Key differences from base GRPOTrainer:
1. Passes pre-encoded CLIP images to model.generate()
2. Uses tokenizer_image_token for proper IMAGE_TOKEN_INDEX handling
3. Model generates text WITH visual context
4. Better quality completions for RL training
"""

import os
import re
import copy
import torch
import logging
from typing import List, Dict, Any, Optional, Union
from contextlib import nullcontext
from trl import GRPOTrainer, GRPOConfig
from trl.trainer.grpo_trainer import unwrap_model_for_generation, pad
from accelerate.utils import gather_object

# Import LaP-Forensic's special tokenization function
from model.llava.mm_utils import tokenizer_image_token
from tools.utils import IMAGE_TOKEN_INDEX

logger = logging.getLogger(__name__)


class LaPGRPOTrainer(GRPOTrainer):
    """
    Custom GRPOTrainer that handles LaP-Forensic's multimodal inputs.
    
    The key modification is passing pre-encoded CLIP images to model.generate(),
    so the model can generate text with proper visual context.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.debug = os.getenv("LAP_FORENSIC_DEBUG", "0") != "0"
        self._gen_call_count = 0
        # Store current batch images for use in _generate
        self._current_batch_images = None
        self._current_batch_indices = None
        self._current_batch_prompts = None
        
        # Completion logging: log every N generation calls
        self.log_completions_every = int(os.getenv("LAP_FORENSIC_LOG_COMPLETIONS_EVERY", "10"))
        self._swanlab = None  # Set externally after trainer creation
        
        # KV cache is now properly supported for multimodal generation
        # after fixing prepare_inputs_labels_for_multimodal to handle DynamicCache
        if hasattr(self, 'generation_config') and self.generation_config is not None:
            if self.generation_config.use_cache is None:
                self.generation_config.use_cache = True
            logger.info(f"Multimodal generation use_cache={self.generation_config.use_cache}")
        
        logger.info("LaPGRPOTrainer initialized (V2 - with KV cache support)")
    
    def _generate_and_score_completions(
        self, inputs: list[dict[str, torch.Tensor | Any]]
    ) -> dict[str, torch.Tensor | Any]:
        """
        Override to extract and store images before generation.
        
        This method is called by TRL to generate completions and compute scores.
        We intercept it to extract images from the batch.
        """
        # Extract pre-encoded images from inputs
        global_enc_images = []
        indices = []
        prompts = []
        
        for x in inputs:
            img = x.get('global_enc_image')
            if img is not None and isinstance(img, torch.Tensor):
                global_enc_images.append(img)
            else:
                global_enc_images.append(None)
            indices.append(x.get('idx'))
            prompts.append(x.get('prompt'))
        
        # Store for use in _generate
        self._current_batch_images = global_enc_images
        self._current_batch_indices = indices
        self._current_batch_prompts = prompts
        
        if self.debug and self._gen_call_count < 5:
            has_images = sum(1 for img in global_enc_images if img is not None)
            num_generations = getattr(self, 'num_generations', 'unknown')
            logger.info(f"LaPGRPOTrainer._generate_and_score_completions: "
                       f"batch_size={len(inputs)}, valid_images={has_images}, num_generations={num_generations}")
            logger.info(f"Expected prompts after TRL expansion: {len(inputs)} (images will be repeated to match)")
            print(f"[GRPO] Original batch: {len(inputs)} samples, {has_images} images, num_generations={num_generations}", flush=True)
        
        # Call parent implementation
        result = super()._generate_and_score_completions(inputs)
        
        # Clear stored images
        self._current_batch_images = None
        self._current_batch_indices = None
        self._current_batch_prompts = None
        
        # Free fragmented GPU memory from generation phase (KV cache, attention intermediates)
        torch.cuda.empty_cache()
        
        return result
    
    def _expand_images_by_prompt_order(
        self,
        expanded_prompts: list,
        images_list: list,
        original_prompts: list,
    ) -> list:
        prompt_to_images = {}
        for prompt, image in zip(original_prompts, images_list):
            prompt_to_images.setdefault(prompt, []).append(image)
        prompt_counters = {prompt: 0 for prompt in prompt_to_images}
        expanded_images = []
        for prompt in expanded_prompts:
            if prompt in prompt_to_images:
                images_for_prompt = prompt_to_images[prompt]
                idx = prompt_counters[prompt] % len(images_for_prompt)
                expanded_images.append(images_for_prompt[idx])
                prompt_counters[prompt] += 1
            else:
                expanded_images.append(None)
        return expanded_images
    
    def _generate(self, prompts: list):
        """
        Override to pass images to model.generate().
        
        This is the core modification - we inject images into the generation process.
        
        CRITICAL: TRL's RepeatSampler repeats each prompt num_generations times.
        We must repeat images to match the expanded prompt count.
        """
        self._gen_call_count += 1
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"
        
        # Copy prompts to avoid modification
        prompts = copy.deepcopy(prompts)
        
        # Check if we have stored images
        images_list = self._current_batch_images
        has_images = images_list is not None and any(img is not None for img in images_list)
        
        # CRITICAL FIX: TRL's RepeatSampler repeats prompts num_generations times
        # We need to repeat images to match the expanded prompt count
        # Example: batch_size=4, num_generations=2 -> 4 images, 8 prompts
        # Each image should be used for 2 consecutive prompts
        num_prompts = len(prompts)
        num_images = len(images_list) if images_list else 0
        
        if has_images and num_prompts > num_images and num_images > 0:
            original_prompts = self._current_batch_prompts
            if original_prompts and len(original_prompts) == num_images:
                images_list = self._expand_images_by_prompt_order(prompts, images_list, original_prompts)
                if self.debug and self._gen_call_count <= 5:
                    logger.info(f"Expanded images by prompt order: {num_images} images -> {len(images_list)} images for {num_prompts} prompts")
            elif num_prompts % num_images == 0:
                repeat_factor = num_prompts // num_images
                images_list = [img for img in images_list for _ in range(repeat_factor)]
                if self.debug and self._gen_call_count <= 5:
                    logger.info(f"Repeated images {repeat_factor}x to match prompts: "
                               f"{num_images} images -> {len(images_list)} images for {num_prompts} prompts")
            else:
                logger.warning(f"Image-prompt count mismatch: {num_images} images, {num_prompts} prompts "
                              f"(not evenly divisible)")
        
        if self.debug and self._gen_call_count <= 3:
            print(f"\n{'='*60}", flush=True)
            print(f"[DEBUG _generate #{self._gen_call_count}] Input Analysis:", flush=True)
            print(f"  has_images={has_images}, num_prompts={num_prompts}, num_images={len(images_list) if images_list else 0}", flush=True)
            # Log first prompt for debugging
            if prompts:
                p0 = prompts[0]
                has_im_start = '<im_start>' in p0
                has_im_end = '<im_end>' in p0
                has_image_token = '<image>' in p0
                print(f"  First prompt tokens: has_im_start={has_im_start}, has_im_end={has_im_end}, has_image={has_image_token}", flush=True)
                print(f"  First prompt preview ({len(p0)} chars):", flush=True)
                print(f"    {p0[:300]}...", flush=True)
            print(f"{'='*60}\n", flush=True)
        
        # If no images, fall back to base implementation
        if not has_images:
            logger.warning("No images available, falling back to text-only generation")
            return super()._generate(prompts)
        
        # Prepare inputs with images
        # CRITICAL: Use tokenizer_image_token to properly handle <image> -> IMAGE_TOKEN_INDEX (-200)
        # Standard tokenizer splits <image> into <, image, > which breaks the model
        
        # Tokenize each prompt individually with image token handling
        input_ids_list = []
        for prompt in prompts:
            # tokenizer_image_token handles <image> -> IMAGE_TOKEN_INDEX
            ids = tokenizer_image_token(prompt, self.processing_class, return_tensors="pt")
            input_ids_list.append(ids)
        
        # Pad to same length (left padding for generation)
        max_len = max(ids.shape[0] for ids in input_ids_list)
        padded_ids = []
        attention_masks = []
        pad_token_id = self.processing_class.pad_token_id or 0
        
        for ids in input_ids_list:
            pad_len = max_len - ids.shape[0]
            if pad_len > 0:
                # Left padding
                padded = torch.cat([
                    torch.full((pad_len,), pad_token_id, dtype=ids.dtype),
                    ids
                ])
                mask = torch.cat([
                    torch.zeros(pad_len, dtype=torch.long),
                    torch.ones(ids.shape[0], dtype=torch.long)
                ])
            else:
                padded = ids
                mask = torch.ones(ids.shape[0], dtype=torch.long)
            padded_ids.append(padded)
            attention_masks.append(mask)
        
        input_ids = torch.stack(padded_ids).to(device)
        attention_mask = torch.stack(attention_masks).to(device)
        
        # CRITICAL FIX: Replace IMAGE_TOKEN_INDEX (-200) with pad_token_id in input_ids
        # for generate(). HuggingFace's RepetitionPenaltyLogitsProcessor uses input_ids
        # to index into logits via scatter — negative indices like -200 cause CUDA
        # "index out of bounds" errors. We pass the original (unsanitized) input_ids
        # as `original_input_ids` so prepare_inputs_for_generation can still use them
        # to locate image token positions for prepare_inputs_labels_for_multimodal.
        sanitized_input_ids = input_ids.clone()
        sanitized_input_ids[sanitized_input_ids == IMAGE_TOKEN_INDEX] = pad_token_id
        
        generate_inputs = {
            "input_ids": sanitized_input_ids,
            "attention_mask": attention_mask,
        }
        
        if self.debug and self._gen_call_count <= 3:
            # Check if IMAGE_TOKEN_INDEX is in input_ids
            has_image_token = (input_ids == IMAGE_TOKEN_INDEX).any().item()
            logger.info(f"Input contains IMAGE_TOKEN_INDEX (-200): {has_image_token}")
            # Check IMAGE_TOKEN_INDEX positions for each sample
            for i in range(min(2, input_ids.shape[0])):
                positions = (input_ids[i] == IMAGE_TOKEN_INDEX).nonzero().squeeze(-1).tolist()
                logger.info(f"Sample {i}: IMAGE_TOKEN positions={positions}, input_ids_len={input_ids[i].shape[0]}")
        
        # Stack images into batch tensor
        # Handle None images by using a placeholder (will be masked)
        normalized_images = []
        reference_image = None
        for img in images_list:
            if img is not None and isinstance(img, torch.Tensor):
                if img.dim() == 4 and img.shape[0] == 1:
                    img = img[0]
                if img.dim() == 3:
                    reference_image = img if reference_image is None else reference_image
                    normalized_images.append(img)
                else:
                    reference_image = img if reference_image is None else reference_image
                    normalized_images.append(img)
            else:
                normalized_images.append(None)
        
        if reference_image is None:
            images_batch = None
            logger.warning("No valid images in batch after normalization")
        else:
            filled_images = []
            for img in normalized_images:
                if img is None:
                    filled_images.append(torch.zeros_like(reference_image))
                else:
                    filled_images.append(img)
            if filled_images and filled_images[0].dim() == 3:
                images_batch = torch.stack(filled_images, dim=0).to(device)
            else:
                images_batch = torch.cat(
                    [img if img.dim() == 4 else img.unsqueeze(0) for img in filled_images],
                    dim=0
                ).to(device)
            model_dtype = next(self.model.parameters()).dtype
            images_batch = images_batch.to(dtype=model_dtype)
            if images_batch.shape[0] != len(prompts):
                logger.error(f"IMAGE-PROMPT MISMATCH: {images_batch.shape[0]} images vs {len(prompts)} prompts!")
                logger.error("This will cause incorrect image-text pairing during generation!")
            if self.debug and self._gen_call_count <= 5:
                match_status = "MATCH" if images_batch.shape[0] == len(prompts) else "MISMATCH"
                logger.info(f"Images batch shape: {images_batch.shape}, dtype: {images_batch.dtype}")
                logger.info(f"Image-Prompt count: {images_batch.shape[0]} images, {len(prompts)} prompts [{match_status}]")
                print(f"[GEN {self._gen_call_count}] Images: {images_batch.shape[0]}, Prompts: {len(prompts)} [{match_status}]", flush=True)
                
                # BATCH-LEVEL DEBUG: Log each sample's image stats and prompt preview
                print(f"\n{'='*60}", flush=True)
                print(f"[BATCH DEBUG] Sample-by-sample image/prompt mapping:", flush=True)
                for i in range(min(4, len(prompts))):
                    img_stats = "N/A"
                    if i < images_batch.shape[0]:
                        img = images_batch[i]
                        img_stats = f"shape={img.shape}, mean={img.mean().item():.4f}, std={img.std().item():.4f}"
                    p_preview = prompts[i][-150:] if len(prompts[i]) > 150 else prompts[i]
                    print(f"  Sample {i}: img={img_stats}", flush=True)
                    print(f"           prompt_end=...{p_preview}", flush=True)
                print(f"{'='*60}\n", flush=True)
        
        # Generate with images
        # CRITICAL: Free cached GPU memory before generation on tight VRAM (e.g. 4090-48G)
        torch.cuda.empty_cache()
        
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            fsdp_context = FSDP.summon_full_params(self.model_wrapped, recurse=False) if self.is_fsdp_enabled else nullcontext()
        except (ImportError, AttributeError):
            fsdp_context = nullcontext()
        
        with (
            unwrap_model_for_generation(
                self.model, self.accelerator, 
                gather_deepspeed3_params=getattr(self.args, 'ds3_gather_for_generation', False)
            ) as unwrapped_model,
            torch.no_grad(),
            fsdp_context,
        ):
            # CRITICAL: Ensure model is in eval mode during generation
            # This affects dropout, batch norm, etc.
            was_training = unwrapped_model.training
            unwrapped_model.eval()
            
            if self.debug and self._gen_call_count <= 3:
                logger.info(f"Model mode: was_training={was_training}, now in eval mode")
                logger.info(f"Generation config: max_new_tokens={self.generation_config.max_new_tokens}, "
                           f"temperature={self.generation_config.temperature}, "
                           f"do_sample={self.generation_config.do_sample}, "
                           f"top_p={self.generation_config.top_p}")
            
            # EXPERIMENTAL: Disable LoRA during generation rollout phase
            # This uses base model weights for generation (which work well)
            # while keeping LoRA for the training phase
            # NOTE: Re-enabled to test if this fixes gibberish output
            disable_lora_for_generation = os.getenv("LAP_FORENSIC_DISABLE_LORA_GEN", "1") == "1"
            has_peft = hasattr(unwrapped_model, 'disable_adapter')
            
            if has_peft and disable_lora_for_generation:
                if self.debug and self._gen_call_count <= 3:
                    logger.info("Disabling LoRA adapters for generation (using base model)")
                adapter_context = unwrapped_model.disable_adapter()
            else:
                adapter_context = nullcontext()
            
            with adapter_context:
                try:
                    # CRITICAL: Pass images and use_left_padding flag to generate
                    # use_left_padding=True tells the model to handle left-padded inputs correctly
                    
                    # DEBUG: Check what type of model we have
                    if self.debug and self._gen_call_count <= 3:
                        logger.info(f"unwrapped_model type: {type(unwrapped_model)}")
                        logger.info(f"unwrapped_model.generate method: {unwrapped_model.generate}")
                        if hasattr(unwrapped_model, 'base_model'):
                            logger.info(f"unwrapped_model.base_model type: {type(unwrapped_model.base_model)}")
                            if hasattr(unwrapped_model.base_model, 'model'):
                                logger.info(f"unwrapped_model.base_model.model type: {type(unwrapped_model.base_model.model)}")
                        if hasattr(unwrapped_model, 'model'):
                            logger.info(f"unwrapped_model.model type: {type(unwrapped_model.model)}")
                        logger.info(f"images_batch is None: {images_batch is None}")
                        if images_batch is not None:
                            logger.info(f"images_batch shape: {images_batch.shape}")
                        
                        # Check prepare_inputs_for_generation
                        if hasattr(unwrapped_model, 'prepare_inputs_for_generation'):
                            logger.info(f"unwrapped_model.prepare_inputs_for_generation: {unwrapped_model.prepare_inputs_for_generation}")
                        if hasattr(unwrapped_model, 'base_model_prepare_inputs_for_generation'):
                            logger.info(f"unwrapped_model.base_model_prepare_inputs_for_generation: {unwrapped_model.base_model_prepare_inputs_for_generation}")
                    
                    # Store original input_ids (with IMAGE_TOKEN_INDEX=-200) on the model
                    # so prepare_inputs_for_generation can use them for image position detection.
                    # Cannot pass as kwarg because generate() rejects unknown model_kwargs.
                    # Must set on the actual LlavaLlamaForCausalLM (not PeftModel wrapper)
                    # because prepare_inputs_for_generation's `self` is LlavaLlamaForCausalLM.
                    base_model = unwrapped_model
                    if hasattr(base_model, 'base_model'):  # PeftModel -> LoraModel
                        base_model = base_model.base_model
                    if hasattr(base_model, 'model'):  # LoraModel -> LlavaLlamaForCausalLM
                        base_model = base_model.model
                    base_model._original_input_ids_for_generation = input_ids
                    prompt_completion_ids = unwrapped_model.generate(
                        **generate_inputs,
                        images=images_batch,  # Pass CLIP-encoded images
                        use_left_padding=True,  # CRITICAL: Handle left-padded inputs for generation
                        generation_config=self.generation_config,
                    )
                    base_model._original_input_ids_for_generation = None
                    
                    # Debug: Log completions (full text, cleaner output)
                    if self.debug and self._gen_call_count <= 3:
                        prompt_length = generate_inputs["input_ids"].size(1)
                        print(f"\n{'='*80}", flush=True)
                        print(f"[BATCH {self._gen_call_count}] Full Completions:", flush=True)
                        print(f"{'='*80}", flush=True)
                        for i in range(min(4, prompt_completion_ids.size(0))):
                            completion_ids = prompt_completion_ids[i, prompt_length:]
                            # Find EOS to get actual completion length
                            eos_positions = (completion_ids == self.eos_token_id).nonzero(as_tuple=True)[0]
                            if len(eos_positions) > 0:
                                actual_len = eos_positions[0].item() + 1
                                completion_ids = completion_ids[:actual_len]
                            completion_text = self.processing_class.decode(completion_ids, skip_special_tokens=False)
                            print(f"\n--- Sample {i} ({len(completion_ids)} tokens) ---", flush=True)
                            print(completion_text, flush=True)
                        print(f"\n{'='*80}\n", flush=True)
                            
                except Exception as e:
                    logger.error(f"Generation failed: {e}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    # Fallback to generation without images
                    if 'base_model' in dir():
                        base_model._original_input_ids_for_generation = None
                    prompt_completion_ids = unwrapped_model.generate(
                        **generate_inputs,
                        generation_config=self.generation_config,
                    )
                finally:
                    # Restore original training mode
                    if was_training:
                        unwrapped_model.train()
        
        # Extract prompt and completion IDs
        prompt_ids = generate_inputs["input_ids"]
        prompt_mask = generate_inputs["attention_mask"]
        prompt_length = prompt_ids.size(1)
        completion_ids = prompt_completion_ids[:, prompt_length:]
        
        # Mask everything after the first EOS token
        is_eos = completion_ids == self.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        
        # Create completion mask
        sequence_indices = torch.arange(completion_ids.size(1), device=device).unsqueeze(0)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).long()
        
        # Convert to lists
        # CRITICAL: Replace IMAGE_TOKEN_INDEX (-200) with a valid token ID for TRL's decode
        # TRL will try to batch_decode these IDs, and -200 will cause IndexError
        # We replace it with pad_token_id (or 0) since it will be skipped anyway
        pad_id = self.processing_class.pad_token_id or 0
        
        prompt_ids_list = []
        for ids, mask in zip(prompt_ids, prompt_mask):
            valid_ids = ids[mask.bool()].tolist()
            # Replace IMAGE_TOKEN_INDEX with pad_id
            valid_ids = [pad_id if x == IMAGE_TOKEN_INDEX else x for x in valid_ids]
            prompt_ids_list.append(valid_ids)
        
        completion_ids_list = [
            ids[mask.bool()].tolist() 
            for ids, mask in zip(completion_ids, completion_mask)
        ]
        
        # Decode completions for TRL
        completions = self.processing_class.batch_decode(completion_ids_list, skip_special_tokens=True)
        
        # Periodic completion logging (in training loop, no extra forward pass)
        if self.log_completions_every > 0 and self._gen_call_count % self.log_completions_every == 0:
            self._log_sample_completions(completions)
        
        # Calculate total completion tokens for metrics
        completion_lengths = torch.tensor([len(ids) for ids in completion_ids_list], device=device)
        agg_completion_lengths = self.accelerator.gather(completion_lengths)
        total_completion_tokens = agg_completion_lengths.sum()
        
        # Log metrics
        prompt_lengths = torch.tensor([len(ids) for ids in prompt_ids_list], device=device)
        agg_prompt_lengths = self.accelerator.gather(prompt_lengths)
        
        if mode == "train":
            self.state.num_input_tokens_seen += (agg_prompt_lengths.sum() + total_completion_tokens).item()
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]
        self._metrics[mode]["completions/mean_length"].append(agg_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(agg_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_length"].append(agg_completion_lengths.float().max().item())
        
        # Check for truncated completions
        eos_and_pad = [self.eos_token_id, self.pad_token_id]
        is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids_list], device=device)
        agg_is_truncated = self.accelerator.gather(is_truncated)
        self._metrics[mode]["completions/clipped_ratio"].append(agg_is_truncated.float().mean().item())
        
        # Return in expected format (matching TRL's _generate return signature)
        # Format: prompt_ids, completion_ids, tool_mask, completions, total_completion_tokens, logprobs, extra_fields
        return (
            prompt_ids_list,
            completion_ids_list,
            None,  # tool_mask
            completions,
            total_completion_tokens,
            None,  # logprobs
            {},    # extra_fields
        )

    def _parse_completion_stats(self, text: str, seg_token: str = "[SEG]") -> dict:
        """Parse structural statistics from a single completion text."""
        seg_count = text.count(seg_token)
        p_open = text.count("<p>")
        p_close = text.count("</p>")
        empty_p = len(re.findall(r'<p>\s*</p>', text))
        valid_units = len(re.findall(
            r'<p>.+?</p>\s*' + re.escape(seg_token), text, re.DOTALL
        ))
        has_steps = bool(re.search(
            r'Step\s*\d+|(?:First|Second|Third|Finally)', text, re.IGNORECASE
        ))
        return {
            "seg_count": seg_count,
            "p_open": p_open,
            "p_close": p_close,
            "empty_p": empty_p,
            "valid_units": valid_units,
            "has_steps": has_steps,
            "token_len": len(text.split()),
        }

    def _log_sample_completions(self, completions: list):
        """
        Log sample completions to SwanLab with structural analysis.
        Called periodically from _generate (in the training loop, no extra forward pass).
        """
        step = getattr(self.state, 'global_step', self._gen_call_count)
        seg_token = "[SEG]"

        # Parse stats for ALL completions (for aggregate metrics)
        all_stats = [self._parse_completion_stats(c, seg_token) for c in completions]

        # Aggregate batch-level structural metrics
        n = max(len(all_stats), 1)
        agg = {
            "completions/pct_has_seg":    sum(1 for s in all_stats if s["seg_count"] > 0) / n,
            "completions/mean_seg_count": sum(s["seg_count"] for s in all_stats) / n,
            "completions/pct_empty_p":    sum(1 for s in all_stats if s["empty_p"] > 0) / n,
            "completions/pct_has_steps":  sum(1 for s in all_stats if s["has_steps"]) / n,
            # valid_format: all SEGs have a proper <p>.+</p> prefix
            "completions/pct_valid_fmt":  sum(
                1 for s in all_stats
                if s["seg_count"] > 0 and s["valid_units"] == s["seg_count"]
            ) / n,
            "completions/mean_word_len":  sum(s["token_len"] for s in all_stats) / n,
        }

        if self._swanlab is not None:
            try:
                import swanlab

                # Log aggregate numeric metrics
                self._swanlab.log(agg, step=step)

                # Log up to 8 sample completions as Text with annotated captions
                samples = list(zip(completions[:8], all_stats[:8]))
                text_list = []
                for i, (c, st) in enumerate(samples):
                    fmt_ok = (st["seg_count"] > 0 and st["valid_units"] == st["seg_count"]
                              and st["empty_p"] == 0)
                    caption = (
                        f"step{step}_s{i} | "
                        f"SEG={st['seg_count']} valid={st['valid_units']} "
                        f"empty_p={st['empty_p']} steps={'Y' if st['has_steps'] else 'N'} "
                        f"words={st['token_len']} fmt={'OK' if fmt_ok else 'BAD'}"
                    )
                    text_list.append(swanlab.Text(c, caption=caption))

                self._swanlab.log({"completions/samples": text_list}, step=step)
            except Exception as e:
                logger.warning(f"SwanLab completion logging failed: {e}")

