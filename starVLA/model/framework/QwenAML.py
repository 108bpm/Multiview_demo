# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Junqiu YU / Fudan University] in [2025]. 
# Design and Merged by [Jinhui YE / HKUST University] in [2025].
"""
Qwen-GR00T Framework
A lightweight implementation that Qwen-VL + Flow-matching head to directly predict continuous actions
Flow-matching header is copyright from GR00T N1.5,
"""
import sys
from pathlib import Path
import math
# Add workspace root to Python path if not already there
_workspace_root = Path(__file__).parent.parent.parent.parent
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))
import os
CHECKPOINT_BASEDIR = os.getenv('CHECKPOINT_BASEDIR', None)
from typing import List
from tqdm import tqdm
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms as TF
import cv2

from starVLA.training.trainer_utils import initialize_overwatch
from deployment.model_server.tools.image_tools import to_pil_preserve

logger = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.modules.action_model.JAT_ActionHeader import get_action_model, FlowmatchingActionHead

from starVLA.training.trainer_utils.trainer_tools import resize_images
from starVLA.model.tools import FRAMEWORK_REGISTRY

from vggt.models.vggt import VGGT
from starVLA.model.modules.projector.QFormer import get_layerwise_qformer
import random

class CrossAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_hidden: int,
        nhead: int = 8,
        dropout: float = 0.0,
        kv_dim: int = 2048
    ):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden if d_hidden is not None else d_model
        self.nhead = nhead
        self.head_dim = self.d_hidden // nhead
        assert self.d_hidden % nhead == 0, "d_hidden must be divisible by nhead"

        # Projections
        self.q_proj = nn.Linear(d_model, self.d_hidden)
        self.k_proj = nn.Linear(kv_dim, self.d_hidden)
        self.v_proj = nn.Linear(kv_dim, self.d_hidden)
        self.out_proj = nn.Linear(self.d_hidden, d_model)

        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_out = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, image_feature: torch.Tensor, spatial_feature: torch.Tensor):
        """
        Args:
            image_feature: (B, N_img, d_model) — Query
            vggt_feature:   (B, N_vggt, kv_dim) — Key and Value

        Returns:
            fused_image_feature: (B, N_img, d_model)
        """
        B, N_img, _ = image_feature.shape
        _, N_spatial, _ = spatial_feature.shape

        # Project to d_hidden
        q = self.q_proj(image_feature)   # (B, N_img, d_hidden)
        k = self.k_proj(spatial_feature)     # (B, N_vggt, d_hidden)
        v = self.v_proj(spatial_feature)     # (B, N_vggt, d_hidden)

        # Reshape for multi-head: (B, N, d_hidden) -> (B, N, nhead, head_dim) -> (B, nhead, N, head_dim)
        q = q.view(B, N_img, self.nhead, self.head_dim).transpose(1, 2)  # (B, nhead, N_img, head_dim)
        k = k.view(B, N_spatial, self.nhead, self.head_dim).transpose(1, 2)  # (B, nhead, N_vggt, head_dim)
        v = v.view(B, N_spatial, self.nhead, self.head_dim).transpose(1, 2)  # (B, nhead, N_vggt, head_dim)

        # Scaled Dot-Product Attention
        scale = self.head_dim ** -0.5
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, nhead, N_img, N_vggt)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout_attn(attn_weights)

        # Weighted sum over values
        attn_output = torch.matmul(attn_weights, v)  # (B, nhead, N_img, head_dim)

        # Concatenate heads and project back
        attn_output = attn_output.transpose(1, 2).contiguous()  # (B, N_img, nhead, head_dim)
        attn_output = attn_output.view(B, N_img, self.d_hidden)  # (B, N_img, d_hidden)

        # Final projection to d_model
        output = self.out_proj(attn_output)  # (B, N_img, d_model)
        output = self.dropout_out(output)

        # Residual connection + LayerNorm
        output = self.norm(image_feature + output)

        return output


class SelfAttention(nn.Module):
    def __init__(self, embed_dim=2560, num_heads=16, dropout=0.1, bias=True):
        """
        Self-Attention Module with Residual Connection and Layer Norm
        
        Args:
            embed_dim (int): 输入 token 的维度
            num_heads (int): 注意力头的数量
            dropout (float): Dropout 概率
            bias (bool): 线性层是否使用 bias
        """
        super(SelfAttention, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # 确保维度可以整除
        assert self.head_dim * num_heads == self.embed_dim, \
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"

        # 【新增】层归一化 LayerNorm
        # Pre-LN 架构：在输入 Attention 之前进行归一化
        self.norm = nn.LayerNorm(embed_dim)

        # 定义 Q, K, V 的线性投影层
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # 输出投影层
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        self.dropout = nn.Dropout(dropout)
        
        # 缩放因子
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor of shape (bs, l, embed_dim)
            mask: Optional attention mask
        
        Returns:
            Output tensor of shape (bs, l, embed_dim) with residual connection
        """
        bs, l, _ = x.shape

        x_normed = self.norm(x)

        # 1. 线性投影得到 Q, K, V (使用归一化后的输入)
        q = self.q_proj(x_normed)
        k = self.k_proj(x_normed)
        v = self.v_proj(x_normed)

        # 2. 多头 reshape 和 transpose
        q = q.view(bs, l, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bs, l, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bs, l, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. 计算 Attention Scores
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=mask, 
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=False
            )
        else:
            # Fallback for older PyTorch versions
            scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
            
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
            
            attn_weights = torch.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            attn_output = torch.matmul(attn_weights, v)

        # 4. 合并多头
        attn_output = attn_output.transpose(1, 2).contiguous().view(bs, l, self.embed_dim)

        # 5. 最终线性投影
        output_proj = self.out_proj(attn_output)
        
        # 应用 Dropout 到投影后的输出 (可选，但推荐)
        output_proj = self.dropout(output_proj)

        out = x + output_proj
        
        return out


def preprocess_images(image_list, target_size, mode='crop'): #  [B，[PLT]]
    batch_images = []
    shapes = set()
    to_tensor = TF.ToTensor()
    # target_size = 518

    # First process all images and collect their shapes
    for imgs in image_list:
        epi_images = []
        img = imgs[0]
        # for img in imgs:
        width, height = img.size

        if mode == "pad":
            # Make the largest dimension 518px while maintaining aspect ratio
            if width >= height:
                new_width = target_size
                new_height = round(height * (new_width / width) / 14) * 14  # Make divisible by 14
            else:
                new_height = target_size
                new_width = round(width * (new_height / height) / 14) * 14  # Make divisible by 14
        else:  # mode == "crop"
            # Original behavior: set width to 518px
            new_width = target_size
            # Calculate height maintaining aspect ratio, divisible by 14
            new_height = round(height * (new_width / width) / 14) * 14

        # Resize with new dimensions (width, height)
        # img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        img = to_tensor(img)  # Convert to tensor (0, 1)

        # Center crop height if it's larger than 518 (only in crop mode)
        if mode == "crop" and new_height > target_size:
            start_y = (new_height - target_size) // 2
            img = img[:, start_y : start_y + target_size, :]

        # For pad mode, pad to make a square of target_size x target_size
        if mode == "pad":
            h_padding = target_size - img.shape[1]
            w_padding = target_size - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                # Pad with white (value=1.0)
                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )

        shapes.add((img.shape[1], img.shape[2]))
        epi_images.append(img)
        batch_images.append(torch.stack(epi_images))

    # Check if we have different shapes
    # In theory our model can also work well with different shapes
    if len(shapes) > 1:
        print(f"Warning: Found images with different shapes: {shapes}")
        # Find maximum dimensions
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)

        # Pad images if necessary
        padded_images = []
        for img in batch_images:
            h_padding = max_height - img.shape[1]
            w_padding = max_width - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )
            padded_images.append(img)
        batch_images = padded_images

    batch_images = torch.stack(batch_images)  # concatenate images

    # Ensure correct shape when single image
    if len(image_list) == 1:
        # Verify shape is (1, C, H, W)
        if batch_images.dim() == 3:
            batch_images = batch_images.unsqueeze(0)
    return batch_images


@FRAMEWORK_REGISTRY.register("QwenAML")
class Qwen_AML(baseframework):
    """
    Multimodal vision-language-action model.

    Components:
      - Qwen2.5 VL interface for fused language/vision token embeddings
      - Layer-wise QFormer for multi-layer feature aggregation
      - DINO encoder for dense multi-view spatial tokens
      - DiT diffusion head for future action sequence modeling

    Focus: Predict future continuous actions conditioned on images + instruction.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        Construct all submodules and cache key configuration values.

        Args:
            config: Hierarchical configuration (OmegaConf/dict) containing framework + trainer sections.
            **kwargs: Reserved for future overrides (unused).
        """
        super().__init__()
        self.config = config
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        # align dims --> we should put them to config or no?
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = self.qwen_vl_interface.model.config.hidden_size

        self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)  # 修复后续引用

        self.future_action_window_size = config.framework.action_model.future_action_window_size
        self.past_action_window_size = config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size
        
        if getattr(self.config.framework, 'spatial_model', None) is not None:
            self.spatial_model = self.get_spatial_model(config)
            self.spatial_projector = self.get_spatial_projector(config)

        if getattr(self.config.framework, 'image_edit_model', None) is not None:
            if not self.config.framework.image_edit_model.read_from_local:
                from starVLA.model.modules.longcat_image_edit_model import LongCatImageEditModel
                if 'LongCat' in config.framework.image_edit_model.model_name_or_path:
                    self.image_edit_model = LongCatImageEditModel.from_pretrained(config.framework.image_edit_model.model_name_or_path, lora_path=config.framework.image_edit_model.lora_path, torch_dtype=torch.bfloat16)
                else:
                    raise NotImplementedError

            self.image_edit_projector = nn.Linear(64, 2560)

            self.spatial_fuser = SelfAttention(embed_dim=config.framework.spatial_projector.output_dim)
            self.view_selector = nn.Sequential(
                nn.Linear(config.framework.spatial_projector.output_dim * 2, config.framework.spatial_projector.output_dim),
                nn.GELU(),
                nn.Linear(config.framework.spatial_projector.output_dim, 1),
                nn.Sigmoid()
            )
            self.spatial_fuser2 = SelfAttention(embed_dim=config.framework.spatial_projector.output_dim)

        if getattr(self.config.framework, 'fuser', None) is None:
            self.config.framework.fuser = {'type':'cross_attention'}
        print(self.config.framework.fuser.type)
        if self.config.framework.fuser.type == 'cross_attention':
            self.fuser = self.get_cross_attention(d_model=config.framework.spatial_projector.output_dim,d_hidden=config.framework.spatial_projector.output_dim,kv_dim=config.framework.spatial_projector.output_dim)
        else:
            raise NotImplementedError

    def get_cross_attention(self, d_model, d_hidden, kv_dim):
        model = CrossAttention(d_model=d_model,d_hidden=d_hidden,kv_dim=kv_dim)
        return model
        
    def get_spatial_model(self, config):
        spatial_model_cfg = config.framework.spatial_model
        if "vggt" in spatial_model_cfg.model_name_or_path:
            self.spatial_type = "vggt"
            spatial_model = VGGT.from_pretrained(spatial_model_cfg.model_name_or_path)
        else:
            raise NotImplementedError
        return spatial_model

    def get_spatial_projector(self, config):
        spatial_projector_cfg = config.framework.spatial_projector
        projector = nn.Linear(config.framework.spatial_model.output_dim, spatial_projector_cfg.output_dim)
        return projector

    def forward_pass_image_edit_model(self, images, prompt=None, render_mv_img=False):
        view_num = getattr(self.config.framework.image_edit_model, 'view_num', 2)
        prompts = ['Rotate the camera view to the left', 'Rotate the camera view to the right']

        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16): 
                outputs = []
                for i in range(view_num):
                    inputs = {
                        "images": images,
                        "prompts": [prompts[i]] * len(images),
                        "generator": torch.Generator("cuda").manual_seed(43),
                        "num_inference_steps": self.config.framework.image_edit_model.num_inference_steps,
                        "guidance_scale": 1.0,
                        "output_type": "latent",
                        "device": 'cuda',
                        'height': 256,
                        'width': 256,
                    }
                    output = self.image_edit_model(**inputs)
                    outputs.append(output)
                output = torch.cat(outputs, dim=1)
        return output        
        
    def forward_pass_VLM(self, batch_images, instructions, mv_feat=None, render_mv_img=False):

        # Step 1: QWenVL input format
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )    
            if getattr(self, 'spatial_model', None) is not None:
                # step 2: encode spatial feature
                with torch.no_grad():
                    if self.spatial_type == "vggt":    
                        spatial_input = preprocess_images(batch_images, batch_images[0][0].size[0]).to(qwen_inputs['pixel_values'].device)   
                        aggregated_tokens_list, ps_idx = self.spatial_model.aggregator(spatial_input)
                    else:
                        raise NotImplementedError
            extra_latents = None
            if getattr(self.config.framework, 'image_edit_model', None) is not None:
                if mv_feat is not None:
                    extra_latents = torch.tensor(np.array(mv_feat), device=qwenvl_outputs.hidden_states[-1].device, dtype=qwenvl_outputs.hidden_states[-1].dtype)
                else:
                    primary_image = [image[0] for image in batch_images]
                    extra_latents = self.forward_pass_image_edit_model(primary_image)

        # step 3: fuse spatial tokens and qwen tokens
        with torch.autocast("cuda", dtype=torch.float32):
            if self.config.framework.fuser.type == 'cross_attention':
                # last_hidden_state: [B, seq_len, H]
                last_hidden = qwenvl_outputs.hidden_states[-1]   # [B, L, H]
                spatial_tokens = None
                if getattr(self, 'spatial_model', None) is not None:
                    if self.spatial_type == "vggt":
                        spatial_tokens = aggregated_tokens_list[-1][:,0,ps_idx:,:]
                    else:
                        raise NotImplementedError

                    spatial_tokens = self.spatial_projector(spatial_tokens)

                if extra_latents is not None:
                    extra_latents = self.image_edit_projector(extra_latents)
                    if spatial_tokens is not None:
                        extra_latents = extra_latents.to(spatial_tokens.dtype)
                        view_num = getattr(self.config.framework.image_edit_model, 'view_num', 2)
                        assert view_num == 2, f"view num should be 2"
                        spatial_token_num = spatial_tokens.shape[1]
                        fused_tokens = torch.cat([spatial_tokens, extra_latents], dim=1)
                        fused_tokens = self.spatial_fuser(fused_tokens)
                        fused_spatial_tokens = fused_tokens[:,:spatial_token_num,:]
                        fused_extra_latents = torch.chunk(fused_tokens[:,spatial_token_num:,:], chunks=view_num, dim=1)
                        gate = self.view_selector(torch.cat(fused_extra_latents, dim=-1))
                        fused_view = gate * fused_extra_latents[0] + (1 - gate) * fused_extra_latents[1]
                        spatial_tokens = torch.cat([fused_spatial_tokens, fused_view], dim=1)
                        spatial_tokens = self.spatial_fuser2(spatial_tokens)

                    else:
                        spatial_tokens = extra_latents
                last_hidden = self.fuser(last_hidden, spatial_tokens)
            
            else:
                raise NotImplementedError

        return last_hidden
            


    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        """

        """
        batch_images = [example["image"] for example in examples]  #  [B，[PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]
        actions = [example["action"] for example in examples]  # label [B， len, 7]
        action_mask = [example["action_mask"] for example in examples] if "action_mask" in examples[0] else None
        state = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]
        use_state = getattr(self.config.framework.action_model, 'use_state', False)
        if not use_state:
            state = None

        mv_feat = [example["mv_feat"] for example in examples] if "mv_feat" in examples[0] else None

        last_hidden = self.forward_pass_VLM(batch_images, instructions, mv_feat=mv_feat)

        # Step 4: Action Expert Forward and Loss
        with torch.autocast("cuda", dtype=torch.float32):
            actions = torch.tensor(
                np.array(actions), device=last_hidden.device, dtype=last_hidden.dtype
            )  # [B, T_full, action_dim]
            actions_target = actions[:, -(self.future_action_window_size+1):, :]  # (B, chunk_len, action_dim)

            repeated_diffusion_steps = (
                self.config.trainer.get("repeated_diffusion_steps", 4) if self.config and self.config.trainer else 4
            )
            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            last_hidden_repeated = last_hidden.repeat(repeated_diffusion_steps, 1, 1)
            
            state_repeated = None
            if state is not None:
                state = torch.tensor(
                    np.array(state), device=last_hidden.device, dtype=last_hidden.dtype
                )
                state_repeated = state.repeat(repeated_diffusion_steps, 1, 1)

            action_mask_repeated = None
            if action_mask is not None:
                action_mask_tensor = torch.tensor(
                    np.array(action_mask), device=last_hidden.device, dtype=torch.bool
                )  # [B, action_dim]
                action_mask_repeated = action_mask_tensor.repeat(repeated_diffusion_steps, 1)  # [B*repeated_diffusion_steps, action_dim]


            action_loss = self.action_model(last_hidden_repeated, actions_target_repeated, state_repeated, action_mask=action_mask_repeated)  # (B, chunk_len, action_dim)

        return {"action_loss": action_loss}

    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict],
        render_mv_img = False,
        **kwargs: str,
    ) -> np.ndarray:
        """
        Steps:
          1. Resize images to training resolution (if specified)
          2. Encode with QwenVL (hidden states retained)
          6. Return normalized action trajectory
        Returns:
            dict:
                normalized_actions (np.ndarray): Shape [B, T, action_dim], diffusion-sampled normalized actions.
        """

        if type(examples) is not list:
            examples = [examples]
        batch_images = [to_pil_preserve(example["image"]) for example in examples]  #  [B，[PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]
    
        state = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]
        use_state = getattr(self.config.framework.action_model, 'use_state', False)
        if not use_state:
            state = None
        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)
        
        last_hidden = self.forward_pass_VLM(batch_images, instructions, render_mv_img=render_mv_img)

        # Step 4: Action Expert Forward
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(last_hidden, state)  # (B, chunk_len, action_dim)

        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}
        



if __name__ == "__main__":
    from omegaconf import OmegaConf
    import debugpy
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="./starVLA/config/training/starvla_cotrain_oxe.yaml", help="Path to YAML config")
    args, clipargs = parser.parse_known_args()

    # debugpy.listen(("0.0.0.0", 10092))
    # print("🔍 Rank 0 waiting for debugger attach on port 10092...")
    # debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)
    # try get model
    cfg.framework.action_model.action_hidden_dim = 2048

    # cfg.framework.qwenvl.base_vlm = f"{CHECKPOINT_BASEDIR}/Florence-2-large"
    

    model: Qwen_GR00T = Qwen_GR00T(cfg)
    print(model)



    # fake sample 
    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    # Create a sample
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16), # action_chunk, action_dim
        "image": [image], # three views
        "lang": "Put all the toys in the child's room - the three board games (two on the bed and one on the table), the two jigsaw puzzles on the table, and the tennis ball on the table - inside the toy box on the table in the child's room.",
        # "state" : np.random.uniform(-1, 1, size=(1, 44)).astype(np.float16), # chunk, state_dim
    }

    batch  = [sample, sample]  # batch size 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    forward_output = model(batch)
    action_loss = forward_output['action_loss']
    print(f"Action Loss: {action_loss.item()}")

    # test predict action
    predict_output = model.predict_action(examples=[sample]) #, state=[batch[0]["state"]]
    normalized_actions = predict_output['normalized_actions']
    print(f"Unnormalized Action: {normalized_actions}")

    # # Advance: try forward model with dataloader
    # # can be fake sample， but here get from dataloader for simpler
    vla_dataset_cfg = cfg.datasets.vla_data
    dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)

    from torch.utils.data import DataLoader
    from starVLA.dataloader.lerobot_datasets import get_vla_dataset, collate_fn
    
    train_dataloader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=1,  # For Debug
        collate_fn=collate_fn,
    )
    # 
    for batch in tqdm(train_dataloader, desc="Processing Batches"):
        batch
        break

    # try get model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model(batch)

    action = model.predict_action(examples=batch)
    print("Finished")
