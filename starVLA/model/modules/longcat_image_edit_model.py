# Copyright 2025 MeiTuan LongCat-Image Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import inspect
import math
import re
from typing import Optional, Union, List, Dict, Any, Tuple
from pathlib import Path
import numpy as np
import PIL
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer, Qwen2VLProcessor
import torch.nn as nn
from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import FromSingleFileMixin
from diffusers.models.autoencoders import AutoencoderKL
from diffusers.models.transformers import LongCatImageTransformer2DModel
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import is_torch_xla_available, logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from dataclasses import dataclass
from typing import List, Union
import os
from PIL import Image
from peft import PeftModel

def split_quotation(prompt_list, quote_pairs=None):
    results = []
    for prompt in prompt_list:
        word_internal_quote_pattern = re.compile(r"[a-zA-Z]+'[a-zA-Z]+")
        matches_word_internal_quote_pattern = word_internal_quote_pattern.findall(prompt)
        mapping_word_internal_quote = []
        temp_prompt = prompt
        for i, word_src in enumerate(set(matches_word_internal_quote_pattern)):
            word_tgt = "longcat_$##$_longcat" * (i + 1)
            temp_prompt = temp_prompt.replace(word_src, word_tgt)
            mapping_word_internal_quote.append([word_src, word_tgt])

        if quote_pairs is None:
            quote_pairs = [("'", "'"), ('"', '"'), ("‘", "’"), ("“", "”")]
        pattern = "|".join([re.escape(q1) + r"[^" + re.escape(q1 + q2) + r"]*?" + re.escape(q2) for q1, q2 in quote_pairs])
        parts = re.split(f"({pattern})", temp_prompt)

        result = []
        for part in parts:
            for word_src, word_tgt in mapping_word_internal_quote:
                part = part.replace(word_tgt, word_src)
            if re.match(pattern, part):
                if len(part):
                    result.append((part, True))
            else:
                if len(part):
                    result.append((part, False))
        results.append(result)
    return results

def prepare_pos_ids(modality_id=0, type="text", start=(0, 0), num_token=None, height=None, width=None):
    if type == "text":
        assert num_token is not None
        pos_ids = torch.zeros(num_token, 3)
        pos_ids[..., 0] = modality_id
        pos_ids[..., 1] = torch.arange(num_token) + start[0]
        pos_ids[..., 2] = torch.arange(num_token) + start[1]
    elif type == "image":
        assert height is not None and width is not None
        pos_ids = torch.zeros(height, width, 3)
        pos_ids[..., 0] = modality_id
        pos_ids[..., 1] = pos_ids[..., 1] + torch.arange(height)[:, None] + start[0]
        pos_ids[..., 2] = pos_ids[..., 2] + torch.arange(width)[None, :] + start[1]
        pos_ids = pos_ids.reshape(height * width, 3)
    else:
        raise KeyError(f'Unknown type {type}, only support "text" or "image".')
    return pos_ids

def calculate_shift(image_seq_len, base_seq_len=256, max_seq_len=4096, base_shift=0.5, max_shift=1.15):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu

def retrieve_timesteps(scheduler, num_inference_steps=None, device=None, timesteps=None, sigmas=None, **kwargs):
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed.")
    if timesteps is not None:
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
    elif sigmas is not None:
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
    return scheduler.timesteps, len(scheduler.timesteps)

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


def calculate_dimensions(target_area, ratio):
    width = math.sqrt(target_area * ratio)
    height = width / ratio
    width = (width // 16 + 1) * 16 if width % 16 != 0 else width
    height = (height // 16 + 1) * 16 if height % 16 != 0 else height
    return int(width), int(height)

# --- 主模型 ---
class LongCatImageEditModel(nn.Module):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: Qwen2_5_VLForConditionalGeneration,
        tokenizer: Qwen2Tokenizer,
        text_processor: Qwen2VLProcessor,
        transformer: LongCatImageTransformer2DModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.vae = vae.eval()
        self.text_encoder = text_encoder.eval()
        self.tokenizer = tokenizer
        self.text_processor = text_processor
        self.transformer = transformer.eval()
        self.scheduler = scheduler

        self.dtype = dtype

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)
        self.image_processor_vl = text_processor.image_processor

        self.image_token = "<|image_pad|>"
        self.prompt_template_encode_prefix = "<|im_start|>system\nAs an image editing expert, first analyze the content and attributes of the input image(s). Then, based on the user's editing instructions, clearly and precisely determine how to modify the given image(s), ensuring that only the specified parts are altered and all other aspects remain consistent with the original(s).<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
        self.prompt_template_encode_suffix = "<|im_end|>\n<|im_start|>assistant\n"
        self.default_sample_size = 128
        self.tokenizer_max_length = 512

    def _encode_prompt_batch(self, prompts: List[str], images: List[Image.Image], device):
        # 批量处理 prompts 和 images
        all_input_ids = []
        all_attention_masks = []
        all_pixel_values = []
        all_image_grid_thw = []

        max_len = 0
        encoded_prompts = []

        for prompt, img in zip(prompts, images):
            # 1. Process image for VL
            vl_inputs = self.image_processor_vl(images=img, return_tensors="pt")
            pixel_values = vl_inputs["pixel_values"]
            image_grid_thw = vl_inputs["image_grid_thw"]

            # 2. Tokenize prompt with quote handling
            tokens_list = split_quotation([prompt])[0]
            all_tokens = []
            for clean_sub, matched in tokens_list:
                if matched:
                    for ch in clean_sub:
                        if ch.strip():
                            sub_tokens = self.tokenizer(ch, add_special_tokens=False)["input_ids"]
                            all_tokens.extend(sub_tokens)
                else:
                    if clean_sub.strip():
                        sub_tokens = self.tokenizer(clean_sub, add_special_tokens=False)["input_ids"]
                        all_tokens.extend(sub_tokens)

            if len(all_tokens) > self.tokenizer_max_length:
                all_tokens = all_tokens[: self.tokenizer_max_length]

            text_tokens_and_mask = self.tokenizer.pad(
                {"input_ids": [all_tokens]},
                max_length=self.tokenizer_max_length,
                padding="max_length",
                return_attention_mask=True,
                return_tensors="pt",
            )

            # 3. Build full input with image placeholders
            text = self.prompt_template_encode_prefix
            merge_length = self.image_processor_vl.merge_size ** 2
            while self.image_token in text:
                num_image_tokens = (image_grid_thw.prod() // merge_length).item()
                text = text.replace(self.image_token, "<|placeholder|>" * num_image_tokens, 1)
            text = text.replace("<|placeholder|>", self.image_token)

            prefix_tokens = self.tokenizer(text, add_special_tokens=False)["input_ids"]
            suffix_tokens = self.tokenizer(self.prompt_template_encode_suffix, add_special_tokens=False)["input_ids"]

            vision_start_token_id = self.tokenizer.convert_tokens_to_ids("<|vision_start|>")
            prefix_len = prefix_tokens.index(vision_start_token_id)

            input_ids = torch.cat([
                torch.tensor(prefix_tokens),
                text_tokens_and_mask.input_ids[0],
                torch.tensor(suffix_tokens)
            ], dim=0)
            attention_mask = torch.cat([
                torch.ones(len(prefix_tokens), dtype=torch.long),
                text_tokens_and_mask.attention_mask[0],
                torch.ones(len(suffix_tokens), dtype=torch.long)
            ], dim=0)

            all_input_ids.append(input_ids)
            all_attention_masks.append(attention_mask)
            all_pixel_values.append(pixel_values.squeeze(0))
            all_image_grid_thw.append(image_grid_thw.squeeze(0))
            max_len = max(max_len, len(input_ids))

        # Pad to max length
        padded_input_ids = []
        padded_attention_masks = []
        for ids, mask in zip(all_input_ids, all_attention_masks):
            pad_len = max_len - len(ids)
            padded_ids = torch.cat([ids, torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=torch.long)])
            padded_mask = torch.cat([mask, torch.zeros(pad_len, dtype=torch.long)])
            padded_input_ids.append(padded_ids)
            padded_attention_masks.append(padded_mask)

        input_ids_batch = torch.stack(padded_input_ids).to(device)
        attention_mask_batch = torch.stack(padded_attention_masks).to(device)
        pixel_values_batch = torch.stack(all_pixel_values).to(device, self.dtype)
        image_grid_thw_batch = torch.stack(all_image_grid_thw).to(device)

        # Encode
        with torch.no_grad():
            outputs = self.text_encoder(
                input_ids=input_ids_batch,
                attention_mask=attention_mask_batch,
                pixel_values=pixel_values_batch,
                image_grid_thw=image_grid_thw_batch,
                output_hidden_states=True,
            )
        last_hidden = outputs.hidden_states[-1]  # [B, L, D]

        # Find actual prompt slice (remove prefix/suffix)
        suffix_len = len(suffix_tokens)
        prompt_embeds = last_hidden[:, prefix_len:-suffix_len, :]

        return prompt_embeds

    def _encode_vae_image_batch(self, image_tensor: torch.Tensor):
        with torch.no_grad():
            dist = self.vae.encode(image_tensor).latent_dist
            latents = dist.mode()  # deterministic
            latents = (latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        return latents

    def _pack_latents(self, latents, height, width):
        B, C = latents.shape[0], latents.shape[1]
        latents = latents.view(B, C, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        return latents.reshape(B, (height // 2) * (width // 2), C * 4)

    def _unpack_latents(self, latents, height, width):
        B, N, C4 = latents.shape
        h_latent = 2 * (int(height) // (self.vae_scale_factor * 2))
        w_latent = 2 * (int(width) // (self.vae_scale_factor * 2))
        C = C4 // 4
        latents = latents.view(B, h_latent // 2, w_latent // 2, C, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        return latents.reshape(B, C, h_latent, w_latent)

    def forward(
        self,
        images: Union[List[Image.Image], torch.Tensor],
        prompts: Union[str, List[str]],
        negative_prompts: Union[str, List[str]] = "",
        num_inference_steps: int = 50,
        guidance_scale: float = 4.5,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: str = "pil",  # "pil" or "tensor"
        device = None,
        width = 512,
        height = 512,
    ) -> Union[List[Image.Image], torch.Tensor]:
        # Normalize inputs
        if isinstance(prompts, str):
            prompts = [prompts]
        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts] * len(prompts)
        elif isinstance(negative_prompts, list) and len(negative_prompts) == 1:
            negative_prompts = negative_prompts * len(prompts)

        batch_size = len(prompts)

        # Handle image input
        if isinstance(images, torch.Tensor):
            assert images.dim() == 4, "images tensor must be [B, C, H, W]"
            assert images.shape[0] == batch_size
            image_tensors = images
            # Assume already resized; use actual H/W
            height, width = images.shape[-2:]
            prompt_images = [self.image_processor.resize(
                Image.fromarray(((img.permute(1,2,0).cpu().numpy() + 1) * 127.5).astype(np.uint8)),
                height // 2, width // 2
            ) for img in images]
        else:
            # List[PIL.Image]
            assert len(images) == batch_size
            orig_sizes = [img.size for img in images]
            ratios = [w / h for w, h in orig_sizes]
            # target_area = 1024 * 1024
            target_area = height * width
            dims = [calculate_dimensions(target_area, r) for r in ratios]
            widths, heights = zip(*dims)

            # Resize main and prompt images
            resized_images = [
                self.image_processor.resize(img, h, w) for img, h, w in zip(images, heights, widths)
            ]
            prompt_images = [
                self.image_processor.resize(img, h // 2, w // 2) for img, h, w in zip(images, heights, widths)
            ]
            image_tensors = torch.cat([
                self.image_processor.preprocess(img, h, w) for img, h, w in zip(resized_images, heights, widths)
            ], dim=0)

        # Encode prompts
        prompt_embeds = self._encode_prompt_batch(prompts, prompt_images, device=device)
        if guidance_scale > 1.0:
            negative_embeds = self._encode_prompt_batch(negative_prompts, prompt_images, device=device)
        else:
            negative_embeds = None

        # Prepare latents
        num_channels_latents = 16
        h_latent = 2 * (int(heights[0]) // (self.vae_scale_factor * 2))  # assume same size
        w_latent = 2 * (int(widths[0]) // (self.vae_scale_factor * 2))

        image_latents = self._encode_vae_image_batch(image_tensors.to(dtype=self.dtype, device=device))
        image_latents_packed = self._pack_latents(image_latents, h_latent, w_latent)

        shape = (batch_size, num_channels_latents, h_latent, w_latent)
        if isinstance(generator, list):
            latents = torch.cat([
                torch.randn((1, *shape[1:]), generator=g, device=device, dtype=self.dtype)
                for g in generator
            ], dim=0)
        else:
            latents = torch.randn(shape, generator=generator, device=device, dtype=self.dtype)
        latents_packed = self._pack_latents(latents, h_latent, w_latent)

        prompt_len = prompt_embeds.shape[1]
        text_ids = prepare_pos_ids(modality_id=0, type="text", start=(0, 0), num_token=prompt_len).to(device)
        latents_ids = prepare_pos_ids(
            modality_id=1, type="image", start=(prompt_len, prompt_len), height=h_latent//2, width=w_latent//2
        ).to(device)
        image_latents_ids = prepare_pos_ids(
            modality_id=2, type="image", start=(prompt_len, prompt_len), height=h_latent//2, width=w_latent//2
        ).to(device, dtype=torch.float64)

        latent_image_ids = torch.cat([latents_ids, image_latents_ids], dim=0)

        # Timesteps
        sigmas = np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps)
        image_seq_len = latents_packed.shape[1]
        mu = calculate_shift(image_seq_len)
        timesteps, _ = retrieve_timesteps(self.scheduler, num_inference_steps, device, sigmas=sigmas, mu=mu)

        # Denoising loop
        for t in timesteps:
            latent_input = torch.cat([latents_packed, image_latents_packed], dim=1)
            timestep = t.expand(latent_input.shape[0]).to(self.dtype)
            noise_pred_text = self.transformer(
                hidden_states=latent_input,
                timestep=timestep / 1000,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
            )[0][:, :image_seq_len]

            if guidance_scale > 1.0:
                noise_pred_uncond = self.transformer(
                    hidden_states=latent_input,
                    timestep=timestep / 1000,
                    encoder_hidden_states=negative_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                )[0][:, :image_seq_len]
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            else:
                noise_pred = noise_pred_text

            latents_packed = self.scheduler.step(noise_pred, t, latents_packed, return_dict=False)[0]

        if output_type == 'latent':
            return latents_packed
        else:
            # Decode
            latents_unpacked = self._unpack_latents(latents_packed, heights[0], widths[0])
            latents_unpacked = (latents_unpacked / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image_decoded = self.vae.decode(latents_unpacked.to(self.vae.dtype), return_dict=False)[0]
            images_pil = self.image_processor.postprocess(image_decoded, output_type="pil")
            return latents_packed, images_pil  # List[PIL.Image.Image]

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: Union[str, os.PathLike],
        lora_path: Union[str, os.PathLike] = None,
        torch_dtype: Optional[torch.dtype] = None,
        device: Union[str, torch.device] = "cpu",
        subfolder: Optional[str] = None,
        **kwargs,
    ):
        """
        Load LongCatImageEditModel from a local directory.

        Expected directory structure:
        pretrained_model_path/
        ├── vae/
        ├── transformer/
        ├── text_encoder/
        ├── tokenizer/
        ├── processor/          # or image_processor/, but Qwen2VLProcessor uses 'processor'
        ├── scheduler/
        └── model_index.json    # optional (not required)

        Args:
            pretrained_model_path (`str` or `os.PathLike`): Path to the model directory.
            torch_dtype (`torch.dtype`, *optional*): Data type to load weights in.
            device (`str` or `torch.device`): Device to load model on.
            subfolder (`str`, *optional*): Not used here, kept for compatibility.
            **kwargs: Additional arguments passed to sub-module loading.

        Returns:
            LongCatImageEditModel instance.
        """
        pretrained_model_path = Path(pretrained_model_path)

        # Resolve device
        if isinstance(device, str):
            device = torch.device(device)

        # 1. Load VAE
        vae = AutoencoderKL.from_pretrained(
            pretrained_model_path / "vae",
            torch_dtype=torch_dtype,
            **kwargs
        )

        # 2. Load Transformer
        transformer = LongCatImageTransformer2DModel.from_pretrained(
            pretrained_model_path / "transformer",
            torch_dtype=torch_dtype,
            **kwargs
        )
        if lora_path is not None:
            transformer = cls.load_lora(lora_path, transformer)
        # 3. Load Text Encoder (Qwen2-VL)
        text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            pretrained_model_path / "text_encoder",
            torch_dtype=torch_dtype,
            **kwargs
        )

        # 4. Load Tokenizer
        tokenizer = Qwen2Tokenizer.from_pretrained(
            pretrained_model_path / "tokenizer",
            **kwargs
        )

        # 5. Load Processor (includes image processor + tokenizer logic)
        # Note: Qwen2VLProcessor expects both 'tokenizer' and 'image_processor' subdirs,
        # but often it's saved as a single 'processor' folder containing config.json and preprocessor_config.json
        processor_path = pretrained_model_path / "text_processor"
        text_processor = Qwen2VLProcessor.from_pretrained(
            processor_path,
            **kwargs
        )

        # 6. Load Scheduler
        # Try scheduler/ first, otherwise use default config
        scheduler_path = pretrained_model_path / "scheduler"
        if scheduler_path.exists():
            scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(scheduler_path)
        else:
            # Create default scheduler (match original pipeline config)
            scheduler = FlowMatchEulerDiscreteScheduler(
                shift=1.0,
                use_karras_sigmas=False,
                sigma_min=0.002,
                sigma_max=1.0,
                sigma_data=0.5,
            )

        # Instantiate model
        model = cls(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_processor=text_processor,
            transformer=transformer,
            scheduler=scheduler,
            dtype=torch_dtype,
        )

        return model
    
    @classmethod
    def load_lora(cls, lora_model_name, transformer):
        transformer = PeftModel.from_pretrained(transformer, lora_model_name)
        transformer = transformer.merge_and_unload()
        return transformer