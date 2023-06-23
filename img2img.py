from diffusers import StableDiffusionImg2ImgPipeline
import torch

device = "cuda" if torch.cuda.is_available() else "mps"
sd_model_path = "danbrown/RevAnimated-v1-2-2"
lora_model_path = "./lora/blindbox_v1_mix.safetensors"

prompt = "detailed, portrait, cartoon, solo, masterpiece"
negative_prompt = "EasyNegative, nsfw, watermark, low quality, worst quality"


class ImageConvert:
    def __init__(self):
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            sd_model_path,
            safety_checker=None,
            requires_safety_checker=False,
        )
        self.pipe = self.pipe.to(device)
        self.pipe.load_lora_weights(lora_model_path)
        self.pipe.enable_xformers_memory_efficient_attention()

    def generate_image(self, image):
        init_image = image.convert("RGB")
        init_image = init_image.resize((512, 512))

        images = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=init_image,
            strength=0.55,
            guidance_scale=7.5,
        ).images
        return images[0]
