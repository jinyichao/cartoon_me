from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline

device = "mps"
sd_model_path = "danbrown/RevAnimated-v1-2-2"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(sd_model_path)
pipe = pipe.to(device)

lora_model_path = "./lora/blindbox_v1_mix.safetensors"
pipe.load_lora_weights(lora_model_path)

init_image = Image.open("test.png").convert("RGB")
init_image = init_image.resize((512, 512))

prompt = "1girl, detailed, portrait, cute, cartoon"
negative_prompt = "nsfw, lowres, watermark"

images = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=init_image,
    strength=0.55,
    guidance_scale=7.5,
).images
images[0].save("output.png")
