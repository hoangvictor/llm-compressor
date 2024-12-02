import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'

import pandas as pd

import torch
from diffusers import StableDiffusion3Pipeline
from custom_blocks import QuantLinear

base_dir = os.path.join(os.path.dirname(__file__), '..', '..')


def run():
    all_results = []
    
    for quant_mode in ['int8', 'fp8', 'fp16']:
        pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16, cache_dir=os.path.join(base_dir, 'data/cache'))
        pipe.to("cuda")

        pipe.transformer.to(memory_format=torch.channels_last)
        pipe.vae.to(memory_format=torch.channels_last)
        torch.set_grad_enabled(False)

        model = pipe.transformer
        all_modules = dict(model.named_modules())

        if quant_mode != 'fp16':
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

        rnt = []
        for _ in range(12):
            st = time.time()
            pipe(
                prompt="A men jumps from a high building",
                height=1024,
                width=1024,
                num_inference_steps=50,
                guidance_scale=7.0
            ).images
            et = time.time()
            if _ >= 2:
                rnt.append(et - st)

        torch.cuda.empty_cache()

        all_results.append([quant_mode, sum(rnt)/len(rnt)])

        del pipe, model
        torch.cuda.empty_cache()

    print(pd.DataFrame(all_results, columns = ['Quant mode', 'Batch running time']))

run()