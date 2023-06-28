# Cartoon me 
This repo makes use of [Stable Diffusion v2](https://github.com/Stability-AI/stablediffusion) and [ReV Animated model](https://civitai.com/models/7371/rev-animated) to generate cartoonized image for a given one using its image-to-image capability.

In addition, by loading the [QR code controlnet model](https://civitai.com/models/90472?modelVersionId=96366), it could further generate fancy images with a readable QR code embedded. 

### Example

<style>
table th:first-of-type {
    width: 33%;
}
table th:nth-of-type(2) {
    width: 33%;
}
table th:nth-of-type(3) {
    width: 33%;
}
</style>
Input Image  |           Generated Image            | Generated Image with QR code 
:---------------------------------:|:------------------------------------:|:-------------------------:
<img src="test.png" width="512"/>  | <img src="example.png" width="512"/> | <img src="qr_example.png" width="512"/>



Live Demo at https://cartoonme.fun/