# Dual-Stream LaP-Forensic Architecture

## Overview

LaP-Forensic 通过引入**重建一致性流 (Consistency Stream)**，在 RGB 语义信息之外显式建模生成图像在潜空间重建后的残差模式，从而强化对高质量深伪内容的定位与解释能力。

核心思想：AI 生成的图像经过 VAE 重建后会产生更大的残差，因为它们已经经过了一次 VAE 编码-解码过程（双重编码效应）。

---

## 架构对比

### Single-Stream Baseline

```
Input Image
    │
    ▼
┌─────────────────┐
│   CLIP ViT-L    │ (Frozen)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  mm_projector   │ (Trainable)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   LLaMA-7B      │ (LoRA)
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
  Text     SAM Mask
Response   Decoder
```

### Dual-Stream LaP-Forensic

```
                    Input Image (I_rgb)
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
    ┌─────────────────┐       ┌─────────────────┐
    │   CLIP ViT-L    │       │   Frozen VAE    │
    │    (Frozen)     │       │  (SD 1.5 VAE)   │
    └────────┬────────┘       └────────┬────────┘
             │                         │
             │                         ▼
             │                ┌─────────────────┐
             │                │ I_rec = VAE(I)  │
             │                └────────┬────────┘
             │                         │
             │                         ▼
             │                ┌─────────────────┐
             │                │ Residual Map    │
             │                │ |I_rgb - I_rec| │
             │                └────────┬────────┘
             │                         │
             ▼                         ▼
    ┌─────────────────┐       ┌─────────────────┐
    │  mm_projector   │       │ CLIP on Residual│
    │   (Trainable)   │       │   (Shared)      │
    └────────┬────────┘       └────────┬────────┘
             │                         │
             │                         ▼
             │                ┌─────────────────┐
             │                │consistency_proj │
             │                │   (Trainable)   │
             │                └────────┬────────┘
             │                         │
             └──────────┬──────────────┘
                        │
                        ▼
              ┌─────────────────┐
              │ Token Concat    │
              │ [img][consist]  │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │   LLaMA-7B      │
              │    (LoRA)       │
              └────────┬────────┘
                       │
                  ┌────┴────┐
                  ▼         ▼
                Text     SAM Mask
              Response   Decoder
```

---

## 核心组件

### 1. DiffusionConsistencyModule

**位置**: `model/consistency_encoder.py`

**功能**: 使用冻结的 SD VAE 生成重建图像和残差图

```python
class DiffusionConsistencyModule(nn.Module):
    """
    Input:  I_rgb [B, 3, H, W] (CLIP normalized)
    Output: I_rec [B, 3, H, W], Residual [B, 3, H, W]
    
    Residual = |I_rgb - VAE.decode(VAE.encode(I_rgb))|
    """
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `vae_path` | `stabilityai/sd-vae-ft-mse` | VAE 模型路径 |
| `dtype` | `bfloat16` | 计算精度 |

### 2. ConsistencyFeatureEncoder

**位置**: `model/consistency_encoder.py`

**功能**: 将残差图编码为 LLM 可理解的 token

| 模式 | 说明 | Token 数量 |
|------|------|-----------|
| `clip` (推荐) | 复用 CLIP 编码器 | 576 |
| `cnn` | 轻量级 CNN | 576 |

### 3. Token Fusion

**位置**: `model/llava/llava_with_region_arch.py`

**策略**: Token 拼接 (Concatenation)

```
Final Input = [text_tokens] + [image_tokens] + [consistency_tokens] + [text_tokens]
                              └────────────────────────────────────┘
                                     视觉信息 (双流融合)
```

---

## 训练配置

### 新增命令行参数

```bash
python scripts/loc_exp/train.py \
    --use_consistency_stream \           # 启用双流
    --vae_path stabilityai/sd-vae-ft-mse \  # VAE 路径
    --consistency_encoder_type clip \    # 编码器类型
    # ... 其他训练参数
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--use_consistency_stream` | bool | False | 是否启用双流 |
| `--vae_path` | str | `stabilityai/sd-vae-ft-mse` | VAE 模型 |
| `--consistency_encoder_type` | str | `clip` | `clip` 或 `cnn` |

### 可训练模块

| 模块 | 训练状态 | 说明 |
|------|----------|------|
| CLIP Vision Tower | ❄️ Frozen | 视觉编码器 |
| VAE | ❄️ Frozen | 重建模块 |
| mm_projector | ✅ Trainable | 视觉投影 |
| **consistency_projector** | ✅ Trainable | **新增** - 一致性投影 |
| LLaMA Layers | 🔧 LoRA | q_proj, v_proj |
| SAM Mask Decoder | ✅ Trainable | 掩码解码 |

---

## 文件变更

| 文件 | 变更类型 | 说明 |
|------|----------|------|
| `model/consistency_encoder.py` | **新增** | VAE 预处理 + 特征编码 |
| `model/llava/llava_with_region_arch.py` | 修改 | 双流融合逻辑 |
| `model/lap.py` | 修改 | 配置选项 |
| `scripts/loc_exp/train.py` | 修改 | CLI 参数 |

---

## 理论依据

### 为什么残差图能检测 AI 生成图像？

1. **双重编码效应**: AI 生成图像已经过一次 VAE 解码，再次编码-解码会产生更大误差
2. **频率特征**: VAE 倾向于平滑高频细节，AI 图像的高频伪影会被放大
3. **语义漂移**: 重建过程中，AI 图像的不自然区域更容易产生语义偏移

### 信噪比验证

```
目标: SNR = mean_error(artifact_region) / mean_error(background) > 1.5
```

---

## 使用示例

### 推理代码

```python
from model.lap import LaPForCausalLM

# 加载模型 (启用双流)
model = LaPForCausalLM.from_pretrained(
    "path/to/checkpoint",
    use_consistency_stream=True,
    vae_path="stabilityai/sd-vae-ft-mse",
    consistency_encoder_type="clip"
)

# 推理
outputs = model.generate(
    input_ids=input_ids,
    images=images,  # 会自动计算残差并融合
    ...
)
```

---

## 后续优化方向

1. **Cross-Attention Fusion**: 替代简单拼接，使用交叉注意力融合双流特征
2. **多尺度残差**: 在 VAE 的多个层级提取残差特征
3. **Diffusion Inversion**: 使用 DDIM inversion 替代纯 VAE 重建
4. **Prompt Engineering**: 在系统提示中加入一致性分析指导语
