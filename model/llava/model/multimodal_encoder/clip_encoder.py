import os
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModel


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, "mm_vision_select_feature", "patch")

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(
                self.vision_tower_name, local_files_only=True
            )
        self.with_region = args.with_region

    def load_model(self):
        print(f"DEBUG: Loading CLIP model from {self.vision_tower_name}...", flush=True)
        self.image_processor = CLIPImageProcessor.from_pretrained(
            self.vision_tower_name, local_files_only=True
        )
        print("DEBUG: Loaded CLIPImageProcessor", flush=True)
        config = CLIPVisionConfig.from_pretrained(self.vision_tower_name, local_files_only=True)
        if os.path.isdir(self.vision_tower_name):
            weights_path = os.path.join(self.vision_tower_name, "model.safetensors")
        else:
            weights_path = hf_hub_download(
                repo_id=self.vision_tower_name, filename="model.safetensors", local_files_only=True
            )
        state_dict = load_file(weights_path)
        vision_state_dict = {k: v for k, v in state_dict.items() if k.startswith("vision_model.")}
        self.vision_tower = CLIPVisionModel(config)
        load_res = self.vision_tower.load_state_dict(vision_state_dict, strict=False)
        if len(load_res.missing_keys) > 0:
            print(
                f"DEBUG: CLIPVisionModel missing_keys={len(load_res.missing_keys)}",
                flush=True,
            )
        if len(load_res.unexpected_keys) > 0:
            print(
                f"DEBUG: CLIPVisionModel unexpected_keys={len(load_res.unexpected_keys)}",
                flush=True,
            )

        print("DEBUG: Loaded CLIPVisionModel", flush=True)
        self.vision_tower.requires_grad_(False)
        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == "patch":
            image_features = image_features[:, 1:]
        elif self.select_feature == "cls_patch":
            image_features = image_features
        else:
            raise ValueError(f"Unexpected select feature: {self.select_feature}")
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if not hasattr(self, "vision_tower") or self.vision_tower is None:
            self.load_model()
            self.vision_tower.to(device=images.device, dtype=images.dtype)
        else:
            embeddings = getattr(getattr(self.vision_tower, "vision_model", None), "embeddings", None)
            position_ids = getattr(embeddings, "position_ids", None)
            if isinstance(position_ids, torch.Tensor) and position_ids.device != images.device:
                embeddings.position_ids = position_ids.to(device=images.device)

        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(
                    image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                    output_hidden_states=True,
                )
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(
                images.to(device=self.device, dtype=self.dtype),
                output_hidden_states=True,
            )
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        torch.cuda.empty_cache()
        if self.with_region:
            return image_features, image_forward_outs
        return image_features, None

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
