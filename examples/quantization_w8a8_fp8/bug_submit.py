import os

os.environ['CUDA_VISIBLE_DEVICES'] = '7'

import torch
from diffusers import StableDiffusion3Pipeline

base_dir = os.path.join(os.path.dirname(__file__), '..', '..')


def run():
    input_prompts = ["A men jumps from a high building"] * 32
    img_save_dir = f'{base_dir}/data/test_generated_img'
    os.makedirs(f'{img_save_dir}', exist_ok=True)

    pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16, cache_dir=os.path.join(base_dir, 'data/cache'))
    pipe.to("cuda")

    torch.set_grad_enabled(False)
    images = pipe(
        prompt="A men jumps from a high building",
        num_images_per_prompt=32,
        height=1024,
        width=1024,
        num_inference_steps=50,
        guidance_scale=7.0
    ).images

    for j in range(len(input_prompts)):
        images[j].save(os.path.join(f'{img_save_dir}', f'{j}.jpg'))

    torch.cuda.empty_cache()
    
    
run()