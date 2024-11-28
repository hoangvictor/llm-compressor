import time
import torch
import json
import os

import pandas as pd

os.environ['CUDA_VISIBLE_DEVICES'] = '7'

import torch
from diffusers import StableDiffusion3Pipeline

from custom_blocks import QuantLinear

base_dir = os.path.join(os.path.dirname(__file__), '..', '..')

save_images = True

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
seed_everything(42)

def partial_quant():
    batch_size = 4
    num_batch = 1
    coco_val_dataset_path = f'{base_dir}/data/coco_all_data_info.json'
    img_save_dir = f'{base_dir}/data/generated_img'

    all_coco_promts_data = json.load(open(coco_val_dataset_path))
    all_coco_images = list(all_coco_promts_data.keys())

    all_results = []
    for quant_mode in ['fp8']: # None, 'int8'
        os.makedirs(f'{img_save_dir}_{quant_mode}', exist_ok=True)

        pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16, cache_dir="/data0/tien/cache")
        pipe.to("cuda")
        pipe.transformer.to(memory_format=torch.channels_last)
        pipe.vae.to(memory_format=torch.channels_last)

        model = pipe.transformer
        all_modules = dict(model.named_modules())

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f'device: {device}')
        torch.set_grad_enabled(False)
        
        if quant_mode is not None:
            print("Quantizing")
            for name, module in all_modules.items():
                if isinstance(module, torch.nn.Linear) and 'transformer_blocks' in name and 'ff' in name:
                    parent_module = all_modules['.'.join(name.split('.')[:-1])]
                    # if fp8_quant_errors[name] < 0.01*int8_quant_errors[name]:
                    #     quant_mode = 'fp8'
                    quant_linear_layer = QuantLinear(module, name, quant_mode=quant_mode)
                    setattr(
                        parent_module, name.split('.')[-1], quant_linear_layer
                    )
            print("Finish quant!")

        # print(model)
        model.to(device)
        rnt = []
        for i in range(0, num_batch*batch_size, batch_size):
            image_ids = all_coco_images[i:i+batch_size]
            input_prompts = [all_coco_promts_data[img] for img in image_ids]

            st = time.time()
            images = pipe(
                prompt=input_prompts,
                negative_prompt="",
                height=512,
                width=512,
                num_inference_steps=28,
                guidance_scale=7.0
            ).images
            et = time.time()
            rnt.append(et - st)

            for j in range(len(image_ids)):
                images[j].save(os.path.join(f'{img_save_dir}_{quant_mode}', image_ids[j]))
            torch.cuda.empty_cache()
        
        rnt = rnt[1:]
        print("Running time:", sum(rnt)/len(rnt))
        all_results.append([quant_mode, sum(rnt)/len(rnt)])

        del pipe, model
        torch.cuda.empty_cache()

    print(pd.DataFrame(all_results, columns = ['Quant mode', 'Batch running time']))

partial_quant()