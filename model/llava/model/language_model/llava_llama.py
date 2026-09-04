#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from typing import List, Optional, Tuple, Union
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import (AutoConfig, AutoModelForCausalLM, LlamaConfig,
                          LlamaForCausalLM, LlamaModel)

from model.llava.llava_with_region_arch import LlavaMetaForCausalLM, LlavaMetaModel


class LlavaConfig(LlamaConfig):
    model_type = "llava_lap"  # Renamed to avoid conflict with transformers built-in llava


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)

        self.model = LlavaLlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Modification for region
        self.num_level_reg_features = 4

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        bboxes: Optional[List[torch.FloatTensor]] = None,
        return_dict: Optional[bool] = None,
        use_left_padding: bool = False,  # RL generation flag
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        (
            input_ids,
            attention_mask,
            past_key_values,
            inputs_embeds,
            labels,
        ) = self.prepare_inputs_labels_for_multimodal(
            input_ids, attention_mask, past_key_values, labels, images, bboxes,
            use_left_padding=use_left_padding
        )
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        if self.training:
            output_hidden_states = outputs.hidden_states
        else:
            output_hidden_states = hidden_states

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=output_hidden_states,  # outputs.hidden_states,
            attentions=outputs.attentions,  # Include attention weights
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        images=None,
        bboxes=None,
        use_left_padding=False,  # RL generation flag
        **kwargs
    ):
#         import os
#         debug = os.environ.get("LAP_FORENSIC_DEBUG", "0") != "0"
#         debug_every = int(os.environ.get("LAP_FORENSIC_DEBUG_EVERY", "100"))  # Print every N calls
        
#         if debug and not hasattr(self, '_prep_call_count'):
#             self._prep_call_count = 0
#         if debug:
#             self._prep_call_count += 1
#             # Print first 5 calls, then every debug_every calls
#             should_print = self._prep_call_count <= 5 or self._prep_call_count % debug_every == 0
#             if should_print:
#                 has_images = images is not None
#                 img_shape = images.shape if has_images else None
#                 has_pkv = past_key_values is not None
#                 # Check what's in kwargs
#                 kwargs_keys = list(kwargs.keys())
#                 # Check if past_key_values is a Cache object with content
#                 pkv_len = 0
#                 if past_key_values is not None:
#                     if hasattr(past_key_values, 'get_seq_length'):
#                         pkv_len = past_key_values.get_seq_length()
#                     elif isinstance(past_key_values, tuple) and len(past_key_values) > 0:
#                         pkv_len = past_key_values[0][0].shape[2] if past_key_values[0] else 0
#                 print(f"[DEBUG prepare_inputs_for_generation #{self._prep_call_count}] has_images={has_images}, img_shape={img_shape}, has_past_key_values={has_pkv}, pkv_len={pkv_len}, kwargs_keys={kwargs_keys}", flush=True)
        
        # CRITICAL: Check if past_key_values has actual content, not just if it's truthy
        # Empty DynamicCache is truthy but should NOT truncate input_ids on first call
        has_cache_content = False
        if past_key_values is not None:
            if hasattr(past_key_values, 'get_seq_length'):
                # DynamicCache or similar
                has_cache_content = past_key_values.get_seq_length() > 0
            elif isinstance(past_key_values, (list, tuple)) and len(past_key_values) > 0:
                # Legacy tuple format
                has_cache_content = True
        
        # CRITICAL FIX: Use original_input_ids (with IMAGE_TOKEN_INDEX = -200 markers)
        # for the model's forward pass, so prepare_inputs_labels_for_multimodal can
        # locate image token positions. The sanitized input_ids (without -200) are used
        # by generate()'s logit processors (e.g., RepetitionPenaltyLogitsProcessor).
        # Stored on model instance because generate() rejects unknown kwargs.
        original_input_ids = getattr(self, '_original_input_ids_for_generation', None)
        
        if has_cache_content:
            input_ids = input_ids[:, -1:]
            # After first generation step, original_input_ids is no longer needed
            # (image tokens are already processed in first forward pass)
            original_input_ids = None

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            # Use original_input_ids (with -200) for model forward if available
            # This enables prepare_inputs_labels_for_multimodal to find image positions
            forward_ids = original_input_ids if original_input_ids is not None else input_ids
            model_inputs = {"input_ids": forward_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": images,
                "bboxes": bboxes,
                "use_left_padding": use_left_padding,
            }
        )
        return model_inputs


# Register with exist_ok=True to avoid errors if already registered
try:
    AutoConfig.register("llava_lap", LlavaConfig)
    AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
except ValueError:
    # Already registered, skip
    pass
