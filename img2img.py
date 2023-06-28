import torch
import io
import pyqrcode
from PIL import Image
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, UniPCMultistepScheduler


device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
sd_model_path = "danbrown/RevAnimated-v1-2-2"
controlnet_path = "DionTimmer/controlnet_qrcode-control_v1p_sd15"

negative_prompt = "(worst quality, low quality:1.4), EasyNegative, nsfw, naked, watermark, angry, sad"


class ImageConvert:
    def __init__(self):
        self.controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch_dtype)

        self.pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            sd_model_path,
            controlnet=self.controlnet,
            torch_dtype=torch_dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )
        self.pipe = self.pipe.to(device)
        if device == "cuda":
            self.pipe.enable_xformers_memory_efficient_attention()
            self.pipe.enable_attention_slicing()
            self.pipe.enable_sequential_cpu_offload()
            self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)

    def generate_image(self, image, prompt, qr_data=None, strength=0.6, scale=2.8):
        init_image = image.convert("RGB")
        if qr_data:
            w, h = 768, 768
            qrobject = pyqrcode.create(qr_data, error="H")
            buffer = io.BytesIO()
            qrobject.png(buffer, scale=20)
            control_image = Image.open(buffer).resize((w, h))
            conditioning_scale = scale
            guidance_scale = 5.0
        else:
            w, h = 512, 512
            control_image = Image.new("RGB", (w, h))
            conditioning_scale = 0.0
            strength = 0.6
            guidance_scale = 7.5
        images = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=init_image.resize((w, h)),
            control_image=control_image,
            strength=strength,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=conditioning_scale,
        ).images
        return images[0]
