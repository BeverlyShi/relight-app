import math
import os
import numpy as np
import torch
import safetensors.torch as sf

from pathlib import Path
from PIL import Image
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from diffusers import AutoencoderKL, UNet2DConditionModel, DPMSolverMultistepScheduler
from diffusers.models.attention_processor import AttnProcessor2_0
from transformers import CLIPTextModel, CLIPTokenizer

MODEL_DIR = Path(os.environ.get("RELIGHT_MODEL_DIR", "/root/autodl-tmp/models"))
SD15_NAME = os.environ.get(
    "RELIGHT_SD15_NAME",
    "/root/autodl-tmp/models/realistic-vision/AI-ModelScope/realistic-vision-v51",
)
IC_LIGHT_PATH = os.environ.get("RELIGHT_IC_LIGHT_PATH", str(MODEL_DIR / "iclight_sd15_fc.safetensors"))
device = torch.device(os.environ.get("RELIGHT_DEVICE", "cuda"))

print("加载模型中...")

tokenizer = CLIPTokenizer.from_pretrained(SD15_NAME, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(SD15_NAME, subfolder="text_encoder")
vae = AutoencoderKL.from_pretrained(SD15_NAME, subfolder="vae")
unet = UNet2DConditionModel.from_pretrained(SD15_NAME, subfolder="unet")

with torch.no_grad():
    new_conv_in = torch.nn.Conv2d(
        8, unet.conv_in.out_channels,
        unet.conv_in.kernel_size,
        unet.conv_in.stride,
        unet.conv_in.padding
    )
    new_conv_in.weight.zero_()
    new_conv_in.weight[:, :4, :, :].copy_(unet.conv_in.weight)
    new_conv_in.bias = unet.conv_in.bias
    unet.conv_in = new_conv_in

unet_original_forward = unet.forward

def hooked_unet_forward(sample, timestep, encoder_hidden_states, **kwargs):
    c_concat = kwargs['cross_attention_kwargs']['concat_conds'].to(sample)
    c_concat = torch.cat([c_concat] * (sample.shape[0] // c_concat.shape[0]), dim=0)
    new_sample = torch.cat([sample, c_concat], dim=1)
    kwargs['cross_attention_kwargs'] = {}
    return unet_original_forward(new_sample, timestep, encoder_hidden_states, **kwargs)

unet.forward = hooked_unet_forward

sd_offset = sf.load_file(IC_LIGHT_PATH)
sd_origin = unet.state_dict()
sd_merged = {k: sd_origin[k] + sd_offset[k] for k in sd_origin.keys()}
unet.load_state_dict(sd_merged, strict=True)
del sd_offset, sd_origin, sd_merged

text_encoder = text_encoder.to(device=device, dtype=torch.float16)
vae = vae.to(device=device, dtype=torch.bfloat16)
unet = unet.to(device=device, dtype=torch.float16)
unet.set_attn_processor(AttnProcessor2_0())
vae.set_attn_processor(AttnProcessor2_0())

scheduler = DPMSolverMultistepScheduler.from_pretrained(
    SD15_NAME, subfolder="scheduler"
)
scheduler.config.algorithm_type = "sde-dpmsolver++"
scheduler.config.use_karras_sigmas = True

t2i_pipe = StableDiffusionPipeline(
    vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
    scheduler=scheduler, safety_checker=None, requires_safety_checker=False,
    feature_extractor=None, image_encoder=None
)
i2i_pipe = StableDiffusionImg2ImgPipeline(
    vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
    scheduler=scheduler, safety_checker=None, requires_safety_checker=False,
    feature_extractor=None, image_encoder=None
)

print("✅ 模型加载完成")


def numpy2pytorch(imgs):
    h = torch.from_numpy(np.stack(imgs, axis=0)).float() / 127.0 - 1.0
    h = h.movedim(-1, 1)
    return h

def pytorch2numpy(imgs):
    results = []
    for x in imgs:
        y = x.movedim(0, -1)
        y = y * 127.5 + 127.5
        y = y.detach().float().cpu().numpy().clip(0, 255).astype(np.uint8)
        results.append(y)
    return results

def _clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))

def make_bg_from_angle(angle_deg, image_width, image_height, brightness=50.0, temperature=5000.0):
    angle_rad = math.radians(angle_deg)
    dx = math.cos(angle_rad)
    dy = -math.sin(angle_rad)
    x_coords = np.linspace(-1, 1, image_width)
    y_coords = np.linspace(1, -1, image_height)
    xx, yy = np.meshgrid(x_coords, y_coords)
    brightness_map = xx * dx + yy * dy
    brightness_map = (brightness_map - brightness_map.min()) / (brightness_map.max() - brightness_map.min())

    # 50 是中性值，不改变默认效果
    brightness_factor = _clamp(brightness / 50.0, 0.2, 2.0)
    brightness_map = np.clip(brightness_map * brightness_factor, 0.0, 1.0)

    # 5000K 是中性值，偏低更暖，偏高更冷
    temperature_norm = _clamp((temperature - 5000.0) / 3000.0, -1.0, 1.0)
    red_scale = 1.0 + max(0.0, -temperature_norm) * 0.22
    blue_scale = 1.0 + max(0.0, temperature_norm) * 0.22

    r = np.clip(brightness_map * red_scale, 0.0, 1.0)
    g = brightness_map
    b = np.clip(brightness_map * blue_scale, 0.0, 1.0)
    bg = np.stack([r, g, b], axis=-1)
    return (bg * 255).astype(np.uint8)

@torch.inference_mode()
def encode_prompt_inner(txt: str):
    max_length = tokenizer.model_max_length
    chunk_length = tokenizer.model_max_length - 2
    id_start = tokenizer.bos_token_id
    id_end = tokenizer.eos_token_id
    id_pad = id_end

    def pad(x, p, i):
        return x[:i] if len(x) >= i else x + [p] * (i - len(x))

    tokens = tokenizer(txt, truncation=False, add_special_tokens=False)["input_ids"]
    chunks = [[id_start] + tokens[i: i + chunk_length] + [id_end] for i in range(0, len(tokens), chunk_length)]
    chunks = [pad(ck, id_pad, max_length) for ck in chunks]
    token_ids = torch.tensor(chunks).to(device=device, dtype=torch.int64)
    conds = text_encoder(token_ids).last_hidden_state
    return conds

@torch.inference_mode()
def encode_prompt_pair(positive_prompt, negative_prompt):
    c = encode_prompt_inner(positive_prompt)
    uc = encode_prompt_inner(negative_prompt)
    max_count = max(len(c), len(uc))
    c = torch.cat([c] * math.ceil(max_count / len(c)), dim=0)[:max_count]
    uc = torch.cat([uc] * math.ceil(max_count / len(uc)), dim=0)[:max_count]
    c = torch.cat([p[None, ...] for p in c], dim=1)
    uc = torch.cat([p[None, ...] for p in uc], dim=1)
    return c, uc

@torch.inference_mode()
def run_relight(
    image: Image.Image,
    angle_deg: float = 0.0,
    brightness: float = 50.0,
    temperature: float = 5000.0,
    prompt: str = "natural lighting",
    negative_prompt: str = "lowres, bad anatomy, bad hands, cropped, worst quality",
    steps: int = 25,
    cfg: float = 2.0,
    seed: int = 12345,
    image_width: int = 512,
    image_height: int = 512,
    highres_scale: float = 1.5,
    highres_denoise: float = 0.3,
    lowres_denoise: float = 0.9,
) -> Image.Image:

    img_np = np.array(image.resize((image_width, image_height))).astype(np.uint8)

    conds, unconds = encode_prompt_pair(
        positive_prompt=prompt + ", best quality",
        negative_prompt=negative_prompt
    )

    rng = torch.Generator(device=device).manual_seed(seed)

    concat_conds = numpy2pytorch([img_np]).to(device=vae.device, dtype=vae.dtype)
    concat_conds = vae.encode(concat_conds).latent_dist.mode() * vae.config.scaling_factor

    input_bg = make_bg_from_angle(
        angle_deg=angle_deg,
        image_width=image_width,
        image_height=image_height,
        brightness=brightness,
        temperature=temperature,
    )
    bg_latent = numpy2pytorch([input_bg]).to(device=vae.device, dtype=vae.dtype)
    bg_latent = vae.encode(bg_latent).latent_dist.mode() * vae.config.scaling_factor

    latents = i2i_pipe(
        image=bg_latent,
        strength=lowres_denoise,
        prompt_embeds=conds,
        negative_prompt_embeds=unconds,
        width=image_width,
        height=image_height,
        num_inference_steps=int(round(steps / lowres_denoise)),
        num_images_per_prompt=1,
        generator=rng,
        output_type='latent',
        guidance_scale=cfg,
        cross_attention_kwargs={'concat_conds': concat_conds},
    ).images.to(vae.dtype) / vae.config.scaling_factor

    pixels = vae.decode(latents).sample
    pixels = pytorch2numpy(pixels)

    hr_width  = int(round(image_width  * highres_scale / 64.0) * 64)
    hr_height = int(round(image_height * highres_scale / 64.0) * 64)
    pixels_hr = [
        np.array(Image.fromarray(p).resize((hr_width, hr_height), Image.LANCZOS))
        for p in pixels
    ]

    fg_hr = np.array(image.resize((hr_width, hr_height))).astype(np.uint8)
    concat_conds_hr = numpy2pytorch([fg_hr]).to(device=vae.device, dtype=vae.dtype)
    concat_conds_hr = vae.encode(concat_conds_hr).latent_dist.mode() * vae.config.scaling_factor

    pixels_tensor = numpy2pytorch(pixels_hr).to(device=vae.device, dtype=vae.dtype)
    latents_hr = vae.encode(pixels_tensor).latent_dist.mode() * vae.config.scaling_factor
    latents_hr = latents_hr.to(device=unet.device, dtype=unet.dtype)

    latents_hr = i2i_pipe(
        image=latents_hr,
        strength=highres_denoise,
        prompt_embeds=conds,
        negative_prompt_embeds=unconds,
        width=hr_width,
        height=hr_height,
        num_inference_steps=int(round(steps / highres_denoise)),
        num_images_per_prompt=1,
        generator=rng,
        output_type='latent',
        guidance_scale=cfg,
        cross_attention_kwargs={'concat_conds': concat_conds_hr},
    ).images.to(vae.dtype) / vae.config.scaling_factor

    pixels_final = vae.decode(latents_hr).sample
    results = pytorch2numpy(pixels_final)
    return Image.fromarray(results[0])
