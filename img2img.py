from diffusers import StableDiffusionImg2ImgPipeline
import torch

device = "cuda" if torch.cuda.is_available() else "mps"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
sd_model_path = "danbrown/RevAnimated-v1-2-2"
# lora_model_path = "./lora/blindbox_v1_mix.safetensors"

prompt = "((best quality)), ((masterpiece)), (detailed)"
negative_prompt = "(worst quality, low quality:1.4), EasyNegative, nsfw, naked, watermark, angry, sad"


class ImageConvert:
    def __init__(self):
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            sd_model_path,
            torch_dtype=torch_dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )
        self.pipe = self.pipe.to(device)
        # self.pipe.load_lora_weights(lora_model_path)
        if device == "cuda":
            self.pipe.enable_xformers_memory_efficient_attention()
            self.pipe.enable_attention_slicing()
            self.pipe.enable_sequential_cpu_offload()

    def generate_image(self, image, strength=0.55):
        init_image = image.convert("RGB")
        init_image = init_image.resize((512, 512))

        images = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=init_image,
            strength=strength,
            guidance_scale=7.5,
        ).images
        return images[0]
