"""
Consistency Encoder Module for Dual-Stream LaP-Forensic.

This module implements the consistency stream preprocessing and feature encoding:
- DiffusionConsistencyModule: Generates reconstruction and residual maps using frozen VAE
- ConsistencyFeatureEncoder: Encodes residual maps into features for LLM fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Literal
from diffusers import AutoencoderKL


class DiffusionConsistencyModule(nn.Module):
    """
    Generates VAE reconstruction and residual maps using a frozen Stable Diffusion VAE.
    
    The residual map captures the difference between the original image and its 
    VAE reconstruction, which tends to be larger for AI-generated images due to
    the double-encoding effect.
    
    Args:
        vae_path: Path or HuggingFace model ID for the VAE (e.g., 'stabilityai/sd-vae-ft-mse')
        device: Device to load the VAE on
        dtype: Data type for VAE computation (default: bfloat16 for memory efficiency)
    """
    
    # ImageNet normalization (used by CLIP)
    IMAGENET_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073])
    IMAGENET_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711])
    
    def __init__(
        self, 
        vae_path: str = "stabilityai/sd-vae-ft-mse",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.vae_path = vae_path
        self.__dict__["_vae"] = None
        self._vae_device: Optional[torch.device] = None
            
        # VAE expects images in [-1, 1] range
        # But CLIP-preprocessed images use ImageNet normalization
        # We need to handle the conversion

    @property
    def vae(self) -> Optional[AutoencoderKL]:
        return self.__dict__.get("_vae", None)

    def _ensure_vae(self, device: torch.device) -> AutoencoderKL:
        vae = self.vae
        if vae is None:
            vae = AutoencoderKL.from_pretrained(
                self.vae_path,
                torch_dtype=self.dtype,
                low_cpu_mem_usage=False,
            )
            vae.requires_grad_(False)
            vae.eval()
            self.__dict__["_vae"] = vae

        if self._vae_device != device:
            vae.to(device=device, dtype=self.dtype)
            self._vae_device = device

        return vae
    
    def _denormalize_from_clip(self, images: torch.Tensor) -> torch.Tensor:
        """Convert from CLIP normalization back to [0, 1] range."""
        mean = self.IMAGENET_MEAN.view(1, 3, 1, 1).to(images.device, images.dtype)
        std = self.IMAGENET_STD.view(1, 3, 1, 1).to(images.device, images.dtype)
        return images * std + mean
    
    def _normalize_for_vae(self, images: torch.Tensor) -> torch.Tensor:
        """Convert from [0, 1] to [-1, 1] range for VAE."""
        return images * 2.0 - 1.0
    
    def _denormalize_from_vae(self, images: torch.Tensor) -> torch.Tensor:
        """Convert from [-1, 1] back to [0, 1] range."""
        return (images + 1.0) / 2.0
    
    def _normalize_for_clip(self, images: torch.Tensor) -> torch.Tensor:
        """Convert from [0, 1] to CLIP normalization."""
        mean = self.IMAGENET_MEAN.view(1, 3, 1, 1).to(images.device, images.dtype)
        std = self.IMAGENET_STD.view(1, 3, 1, 1).to(images.device, images.dtype)
        return (images - mean) / std
    
    @torch.no_grad()
    def forward(
        self, 
        images: torch.Tensor,
        return_reconstruction: bool = True
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """
        Compute VAE reconstruction and residual map.
        
        Args:
            images: Input images with CLIP normalization [B, 3, H, W]
            return_reconstruction: Whether to return the reconstructed image
            
        Returns:
            Tuple of:
            - reconstructed_images: VAE-reconstructed images (same normalization as input)
            - residual_map: Absolute difference map [B, 3, H, W] or [B, 1, H, W] if grayscale
        """
        original_dtype = images.dtype
        images = images.to(self.dtype)
        vae = self._ensure_vae(images.device)
        
        # Handle size: VAE works best with sizes divisible by 8
        _, _, H, W = images.shape
        need_resize = (H % 8 != 0) or (W % 8 != 0)
        
        if need_resize:
            # Pad to nearest multiple of 8
            new_H = ((H + 7) // 8) * 8
            new_W = ((W + 7) // 8) * 8
            images_padded = F.pad(images, (0, new_W - W, 0, new_H - H), mode='reflect')
        else:
            images_padded = images
            new_H, new_W = H, W
        
        # Convert from CLIP normalization to VAE range [-1, 1]
        images_01 = self._denormalize_from_clip(images_padded)
        images_01 = torch.clamp(images_01, 0, 1)  # Ensure valid range
        images_vae = self._normalize_for_vae(images_01)
        
        # Encode and decode through VAE
        latents = vae.encode(images_vae).latent_dist.sample()
        reconstructed_vae = vae.decode(latents).sample
        
        # Convert back to [0, 1] range
        reconstructed_01 = self._denormalize_from_vae(reconstructed_vae)
        reconstructed_01 = torch.clamp(reconstructed_01, 0, 1)
        
        # Remove padding if applied
        if need_resize:
            reconstructed_01 = reconstructed_01[:, :, :H, :W]
            images_01 = images_01[:, :, :H, :W]
        
        # Compute residual map in [0, 1] space (more interpretable)
        residual_map = torch.abs(images_01 - reconstructed_01)
        
        # Offload VAE back to CPU to free GPU memory for training phase
        # VAE will be lazy-loaded back to GPU on next call (~30ms penalty)
        if hasattr(self, '_vae') and self._vae is not None:
            self._vae.to(device='cpu')
            self._vae_device = torch.device('cpu')
            torch.cuda.empty_cache()
        
        # Convert residual map to CLIP normalization for encoder compatibility
        # We use the residual as a 3-channel "image" for the encoder
        residual_map_normalized = self._normalize_for_clip(residual_map)
        
        if return_reconstruction:
            reconstructed_clip = self._normalize_for_clip(reconstructed_01)
            return reconstructed_clip.to(original_dtype), residual_map_normalized.to(original_dtype)
        
        return None, residual_map_normalized.to(original_dtype)


class ConsistencyFeatureEncoder(nn.Module):
    """
    Encodes residual maps into features compatible with LLM input.
    
    Supports two encoder types:
    - 'clip': Uses a shared/separate CLIP encoder for feature extraction
    - 'cnn': Uses a lightweight CNN encoder
    
    Args:
        encoder_type: Type of encoder ('clip' or 'cnn')
        clip_model_name: CLIP model name (if using 'clip' encoder type)
        hidden_dim: Output hidden dimension (should match LLM hidden size)
        use_shared_clip: If True and encoder_type='clip', expects external CLIP features
    """
    
    def __init__(
        self,
        encoder_type: Literal["clip", "cnn"] = "clip",
        clip_model_name: str = "openai/clip-vit-large-patch14-336",
        hidden_dim: int = 1024,  # CLIP ViT-L hidden dim
        use_shared_clip: bool = True
    ):
        super().__init__()
        self.encoder_type = encoder_type
        self.use_shared_clip = use_shared_clip
        self.hidden_dim = hidden_dim
        self._clip_model_name = clip_model_name
        self.encoder = None
        
        if encoder_type == "cnn":
            # Lightweight CNN encoder for residual maps
            self.encoder = nn.Sequential(
                # Input: [B, 3, 336, 336] -> [B, 64, 168, 168]
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # -> [B, 64, 84, 84]
                
                # Residual-style blocks
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),  # -> [B, 128, 42, 42]
                
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),  # -> [B, 256, 21, 21]
                
                nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),  # -> [B, 512, 21, 21]
                
                # Adaptive pooling to get fixed spatial size
                nn.AdaptiveAvgPool2d((24, 24)),  # -> [B, 512, 24, 24]
            )
            
            # Project to hidden_dim and reshape to token sequence
            self.feature_proj = nn.Sequential(
                nn.Linear(512, hidden_dim),
                nn.GELU(),
            )
            
        elif encoder_type == "clip" and not use_shared_clip:
            self.encoder = None
        
        # For shared CLIP, no encoder is needed - features come from external source
    
    def _ensure_clip_encoder(self, device: torch.device) -> nn.Module:
        if self.encoder is None:
            from transformers import CLIPVisionModel

            encoder = CLIPVisionModel.from_pretrained(
                self._clip_model_name,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=False,
            )
            encoder.requires_grad_(False)
            encoder.eval()
            self.encoder = encoder

        self.encoder.to(device=device)
        return self.encoder

    def forward(
        self, 
        residual_map: torch.Tensor,
        shared_clip_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode residual map into feature tokens.
        
        Args:
            residual_map: Residual map [B, 3, H, W] with CLIP normalization
            shared_clip_features: Pre-computed CLIP features if use_shared_clip=True
            
        Returns:
            features: Feature tokens [B, N_tokens, hidden_dim]
        """
        if self.encoder_type == "clip" and self.use_shared_clip:
            if shared_clip_features is None:
                raise ValueError("shared_clip_features required when use_shared_clip=True")
            return shared_clip_features
        
        elif self.encoder_type == "clip":
            # Use separate CLIP encoder
            encoder = self._ensure_clip_encoder(residual_map.device)
            outputs = encoder(residual_map, output_hidden_states=True)
            # Use second-to-last hidden state (common practice)
            features = outputs.hidden_states[-2]
            return features[:, 1:]  # Remove CLS token, keep patch tokens
        
        else:  # CNN encoder
            # Extract spatial features
            spatial_features = self.encoder(residual_map)  # [B, 512, 24, 24]
            B, C, H, W = spatial_features.shape
            
            # Reshape to token sequence: [B, H*W, C]
            features = spatial_features.flatten(2).transpose(1, 2)  # [B, 576, 512]
            
            # Project to hidden_dim
            features = self.feature_proj(features)  # [B, 576, hidden_dim]
            
            return features


class DualStreamConsistencyEncoder(nn.Module):
    """
    Complete dual-stream consistency encoder combining VAE preprocessing and feature encoding.
    
    This is a convenience wrapper that combines DiffusionConsistencyModule and 
    ConsistencyFeatureEncoder for easy integration.
    
    Args:
        vae_path: Path to VAE model
        encoder_type: Type of feature encoder ('clip' or 'cnn')
        hidden_dim: Output feature dimension
        use_shared_clip: Whether to use shared CLIP encoder
    """
    
    def __init__(
        self,
        vae_path: str = "stabilityai/sd-vae-ft-mse",
        encoder_type: Literal["clip", "cnn"] = "clip",
        hidden_dim: int = 1024,
        use_shared_clip: bool = True
    ):
        super().__init__()
        
        self.consistency_module = DiffusionConsistencyModule(vae_path=vae_path)
        self.feature_encoder = ConsistencyFeatureEncoder(
            encoder_type=encoder_type,
            hidden_dim=hidden_dim,
            use_shared_clip=use_shared_clip
        )
        
        self.use_shared_clip = use_shared_clip
    
    def forward(
        self, 
        images: torch.Tensor,
        vision_tower: Optional[nn.Module] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward pass: VAE reconstruction -> residual map -> feature encoding.
        
        Args:
            images: Input images with CLIP normalization [B, 3, H, W]
            vision_tower: Shared CLIP vision tower (required if use_shared_clip=True)
            
        Returns:
            Tuple of:
            - consistency_features: Encoded features [B, N_tokens, hidden_dim]
            - residual_map: Raw residual map for visualization [B, 3, H, W]
        """
        #### Generate residual map
        _, residual_map = self.consistency_module(images, return_reconstruction=False)
        
        # Encode features
        if self.use_shared_clip and vision_tower is not None:
            #### Use shared CLIP encoder on residual map
            shared_features, _ = vision_tower(residual_map)
            consistency_features = self.feature_encoder(residual_map, shared_features)
        else:
            consistency_features = self.feature_encoder(residual_map)
        
        return consistency_features, residual_map
