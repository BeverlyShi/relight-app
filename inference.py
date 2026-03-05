import math
import numpy as np
import torch
import safetensors.torch as sf

from pathlib import Path
from PIL import Image
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from diffusers import AutoencoderKL, UNet2DConditionModel, DPMSolverMultistepScheduler
from diffusers.models.attention_processor import AttnProcessor2_0
from transformers import CLIPTextModel, CLIPTokenizer, AutoModelForImageSegmentation
from torchvision import transforms

# ── 路径配置 ──────────────────────────────────────
MODEL_DIR = Path("/root/autodl-tmp/models")
SD15_NAME = str(MODEL_DIR / "realistic-vision/AI-ModelScope/realistic-vision-v51")
IC_LIGHT_PATH = str(MODEL_DIR / "iclight_sd15_fc.safetensors")
BIREFNET_PATH = str(MODEL_DIR / "BiRefNet")
device = torch.device("cuda")

# ── 加载 IC-Light 模型 ────────────────────────────
print("加载 IC-Light 模型中...")

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

scheduler = DPMSolverMultistepScheduler.from_pretrained(SD15_NAME, subfolder="scheduler")
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

print("✅ IC-Light 模型加载完成")

# ── 加载 BiRefNet ─────────────────────────────────
print("加载 BiRefNet 模型中...")

birefnet = AutoModelForImageSegmentation.from_pretrained(
    BIREFNET_PATH, trust_remote_code=True
)
birefnet = birefnet.to(device=device, dtype=torch.float32)
birefnet.eval()

birefnet_transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

print("✅ BiRefNet 模型加载完成")


# ── 工具函数 ──────────────────────────────────────

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

def make_bg_from_angle(angle_deg, image_width, image_height):
    angle_rad = math.radians(angle_deg)
    dx = math.cos(angle_rad)
    dy = -math.sin(angle_rad)
    x_coords = np.linspace(-1, 1, image_width)
    y_coords = np.linspace(1, -1, image_height)
    xx, yy = np.meshgrid(x_coords, y_coords)
    brightness = xx * dx + yy * dy
    brightness = (brightness - brightness.min()) / (brightness.max() - brightness.min())
    brightness = (brightness * 255).astype(np.uint8)
    return np.stack([brightness] * 3, axis=-1)

@torch.inference_mode()
def segment_foreground(image: Image.Image):
    orig_w, orig_h = image.size
    input_tensor = birefnet_transform(image).unsqueeze(0).to(device)
    preds = birefnet(input_tensor)[-1].sigmoid()
    mask = preds[0].squeeze().cpu().numpy()
    mask_img = Image.fromarray((mask * 255).astype(np.uint8)).resize((orig_w, orig_h), Image.LANCZOS)
    mask_np = np.array(mask_img)
    fg_rgba = image.convert("RGBA")
    fg_rgba.putalpha(mask_img)
    return fg_rgba, mask_np

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

    # 1. 保存原始背景
    original_image = image.copy().resize((image_width, image_height))

    # 2. BiRefNet 分割前景
    fg_rgba, mask_np = segment_foreground(image)
    fg_rgb = fg_rgba.convert("RGB").resize((image_width, image_height))
    mask_resized = Image.fromarray(mask_np).resize((image_width, image_height), Image.LANCZOS)
    mask_np_resized = np.array(mask_resized) / 255.0

    # 3. IC-Light 对前景打光
    img_np = np.array(fg_rgb).astype(np.uint8)

    conds, unconds = encode_prompt_pair(
        positive_prompt=prompt + ", best quality",
        negative_prompt=negative_prompt
    )

    rng = torch.Generator(device=device).manual_seed(seed)

    concat_conds = numpy2pytorch([img_np]).to(device=vae.device, dtype=vae.dtype)
    concat_conds = vae.encode(concat_conds).latent_dist.mode() * vae.config.scaling_factor

    input_bg = make_bg_from_angle(angle_deg, image_width, image_height)
    bg_latent = numpy2pytorch([input_bg]).to(device=vae.device, dtype=vae.dtype)
    bg_latent = vae.encode(bg_latent).latent_dist.mode() * vae.config.scaling_factor

    # 第一阶段
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

    # 放大
    hr_width  = int(round(image_width  * highres_scale / 64.0) * 64)
    hr_height = int(round(image_height * highres_scale / 64.0) * 64)
    pixels_hr = [
        np.array(Image.fromarray(p).resize((hr_width, hr_height), Image.LANCZOS))
        for p in pixels
    ]

    fg_hr = np.array(fg_rgb.resize((hr_width, hr_height))).astype(np.uint8)
    concat_conds_hr = numpy2pytorch([fg_hr]).to(device=vae.device, dtype=vae.dtype)
    concat_conds_hr = vae.encode(concat_conds_hr).latent_dist.mode() * vae.config.scaling_factor

    pixels_tensor = numpy2pytorch(pixels_hr).to(device=vae.device, dtype=vae.dtype)
    latents_hr = vae.encode(pixels_tensor).latent_dist.mode() * vae.config.scaling_factor
    latents_hr = latents_hr.to(device=unet.device, dtype=unet.dtype)

    # 第二阶段
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
    relit_image = Image.fromarray(results[0])

    # 4. 合成回原始背景
    relit_resized = relit_image.resize((image_width, image_height))
    mask_3ch = np.stack([mask_np_resized] * 3, axis=-1)
    orig_np = np.array(original_image).astype(np.float32)
    relit_np = np.array(relit_resized).astype(np.float32)
    composite = relit_np * mask_3ch + orig_np * (1 - mask_3ch)
    composite = composite.clip(0, 255).astype(np.uint8)

    return Image.fromarray(composite)