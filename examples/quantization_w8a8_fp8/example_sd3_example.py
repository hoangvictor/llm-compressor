import time
import torch
import logging
import json

import os
import torch
import torch.nn as nn
from diffusers import StableDiffusion3Pipeline
from vllm.model_executor.layers.quantization.utils.w8a8_utils import apply_fp8_linear, apply_int8_linear

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
quant_mode = 'int8'

def calculate_adjusted_scale_factor(qvals: torch.Tensor):
    scales = []
    exp_vals = [-9, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8]
    for i in range(len(exp_vals)):
        if exp_vals[i] == -9:
            min_val = 0
            max_val = exp_vals[i+1]
        elif exp_vals[i] == 8:
            min_val = exp_vals[i]
            max_val = 448
        else:
            min_val = 2**(exp_vals[i])
            max_val = 2**(exp_vals[i+1])

        considered_vals = qvals[(qvals >= min_val) & (qvals < max_val)]
        if len(considered_vals) > 1:
            quant_range = max_val - min_val
            true_range = considered_vals.max() - considered_vals.min()
            s = true_range/quant_range
        else:
            s = 1
        scales.append(s)
    return scales


def adjust_local_scales(local_scales: list[torch.Tensor], vals: torch.Tensor, step: int):
    exp_vals = torch.tensor([-9, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.float32)
    exp_bounds = 2 ** exp_vals
    exp_bounds[0] = 0  # Special case for -9 to start at 0
    exp_bounds = torch.cat((exp_bounds, torch.tensor([448.0])))  # Append the upper bound
    
    for i, scale in enumerate(local_scales):
        bounds_min = exp_bounds[:-1]
        bounds_max = exp_bounds[1:]
        scale_mask = scale != 1  # Mask for scales not equal to 1

        for j in torch.where(scale_mask)[0]:  # Only process non-1 scales
            min_val, max_val = bounds_min[j], bounds_max[j]
            mask = (vals[:, i*step:(i+1)*step] >= min_val) & (vals[:, i*step:(i+1)*step] < max_val)
            vals[:, i*step:(i+1)*step][mask] *= scale[j]

    return vals


def calculate_local_scales(qvals: torch.Tensor):
    2**(torch.floor(torch.log2(qvals))-7)


collected_activations = {}
class CollectInputsLinear(nn.Linear):
    def __init__(
        self,
        module: nn.Linear,
        old_module_name: str = None,
        do_classifier_free_guidance: bool = True
    ):
        super().__init__(module.in_features, module.out_features, 'bias' in module._parameters)
        self.forward_call = 0
        self.old_module_name = old_module_name
        self.weight = module.weight
        self.in_features = module.in_features
        self.out_features = module.out_features
        self.do_classifier_free_guidance = do_classifier_free_guidance
        if 'bias' in module._parameters:
            self.bias = module.bias
        else:
            self.register_parameter("bias", None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if len(input.shape) == 2:
            current_max = torch.max(input).reshape(1,)
        else:
            current_max = torch.max(torch.max(input.reshape(input.shape[0], -1, input.shape[-1]), dim=0)[0], dim=1)[0]
        key_name = self.old_module_name
        if key_name not in collected_activations:
            collected_activations[key_name] = current_max
        else:
            collected_activations[key_name] = torch.max(
                current_max, 
                collected_activations[key_name]
            )
        self.forward_call += 1
        return super().forward(input)


class FP8Linear(nn.Module):

    def __init__(
        self,
        module: nn.Linear,
        old_module_name: str = None,
        quant_mode: str = 'fp8'
    ):
        super().__init__()        
        self.local_scales = []

        max_abs_val = torch.max(torch.abs(module.weight.T), axis=0).values.to(torch.float32)
        # max_abs_exp_int = torch.log2(max_abs_val).to(int)
        # max_abs_exp_int_list, max_abs_exp_int_count = max_abs_exp_int.unique(return_counts=True)
        # most_freq_max_abs_exp_int = max_abs_exp_int_list[torch.argmax(max_abs_exp_int_count)]
        # additional_scale = 2**(most_freq_max_abs_exp_int)

        finfo = torch.finfo(torch.float8_e4m3fn)
        
        assert quant_mode in ['fp8', 'int8']
        self.quant_mode = quant_mode
        self.old_module_name = old_module_name
        print(f"Module {old_module_name}")
        
        self.input_scale = collected_activations[self.old_module_name].to(torch.float32) # None
        min_fp8_val = finfo.min
        max_fp8_val = finfo.max # * additional_scale
        self.fp8_weight_scale = (max_abs_val / max_fp8_val).clamp(min=1e-12)
        self.fp8_weight_scale = self.fp8_weight_scale.contiguous()
        self.fp8_weight = (module.weight.T / self.fp8_weight_scale).clamp(min=min_fp8_val, max=max_fp8_val)

        # for col_idx in range(0,self.fp8_weight.shape[1], 128):
        #     qvals = torch.abs(self.fp8_weight[:, col_idx])
        #     self.local_scales.append(calculate_adjusted_scale_factor(qvals))

        # self.local_scales = torch.Tensor(self.local_scales)
        # self.fp8_weight = adjust_local_scales(self.local_scales, self.fp8_weight, 128)
        self.fp8_weight = self.fp8_weight.to(torch.float8_e4m3fn)

        # import matplotlib.pyplot as plt
        # a = module.weight.T[0]
        # plt.hist(a[a<0].cpu().detach().numpy(), bins=100)
        # plt.savefig('fig<0.png')

        # plt.clf()
        # plt.hist(a[a>0].cpu().detach().numpy(), bins=100)
        # plt.savefig('fig>0.png')
        
        iinfo = torch.iinfo(torch.int8)
        self.int8_weight_scale = (max_abs_val / iinfo.max).clamp(min=1e-12)
        self.int8_weight_scale = self.int8_weight_scale.contiguous()
        self.int8_weight = (module.weight.T / self.int8_weight_scale).clamp(min=iinfo.min, max=iinfo.max).to(torch.int8)

        # if torch.abs(module.weight.T-(self.fp8_weight.to(torch.float32)*self.fp8_weight_scale)).sum() > torch.abs(module.weight.T-(self.int8_weight*self.int8_weight_scale)).sum():
        #     self.quant_mode = 'int8'

        if module._parameters['bias'] is not None:
            self.bias = module.bias
        else:
            self.register_parameter("bias", None)

    @torch.no_grad()
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input_2d = input.reshape(-1, input.shape[-1])
        output_shape = [*input.shape[:-1], self.fp8_weight.shape[1]]
        scales = torch.concat(input.shape[0]*[self.input_scale]).reshape(-1,1) if self.input_scale is not None else None
        if self.quant_mode == 'fp8':
            output = apply_fp8_linear(input_2d, self.fp8_weight, input_scale=scales, weight_scale=self.fp8_weight_scale, bias=self.bias, use_per_token_if_dynamic=True, module_name=self.old_module_name)
        elif self.quant_mode == 'int8':
            output = apply_int8_linear(input_2d, self.int8_weight, input_scale=scales, weight_scale=self.int8_weight_scale, bias=self.bias)
        else:
            raise Exception(f"The forward pass of this quant mode `{self.quant_mode}` is not implemented yet.")
        return output.view(*output_shape)

    def extra_repr(self) -> str:
        return f"in_features={self.fp8_weight.shape[0]}, out_features={self.fp8_weight.shape[1]}, bias={self.bias is not None}, quant_mode={self.quant_mode}"


def example():
    pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16, cache_dir="/data0/tien/cache")
    pipe.to("cuda")

    # if torch.cuda.is_available():
    #     torch.backends.cudnn.benchmark = True
    #     torch.backends.cudnn.deterministic = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'device: {device}')
    
    print("Inserting activations quantizers ...")
    model = pipe.transformer

    print("Quantizing ...")

    print("Finish quant!")
    logging.info(model)

    model.to(device)
    model.half()
    torch.set_grad_enabled(False)
    
    old_module = model.transformer_blocks[0].ff.net[-1]
    fp8_linear_module = FP8Linear(old_module)
    old_module.eval()
    fp8_linear_module.eval()
    print(fp8_linear_module)

    num_runs = 1
    inputs = [torch.rand([512,6144], dtype=torch.float16).to('cuda') for _ in range(num_runs)]

    output1, output2 = 0, 0
    total_time_custom_linear = 0
    total_time_original_linear = 0
    quant_err = torch.zeros_like(old_module(inputs[0]))
    for _ in range(num_runs):
        with torch.no_grad():
            st = time.time()
            output2 = old_module(inputs[_])
            total_time_original_linear += time.time() - st

    for _ in range(num_runs):
        # qinput, x_scale = ops.scaled_fp8_quant(
        #     inputs[_],
        #     None,
        #     scale_ub=None)
        with torch.no_grad():
            st = time.time()
            output1 = fp8_linear_module(inputs[_]) #, x_scale.contiguous())
            total_time_custom_linear += time.time() - st

    print('Average calculation time of original Linear layer:', total_time_original_linear/num_runs)
    print('Average calculation time of custom Linear layer:', total_time_custom_linear/num_runs)

    print(output1, output2)


def quant():
    batch_size = 32
    coco_val_dataset_path = '/data0/tien/llm-compressor/data/coco_all_data_info.json'
    img_save_dir = '/data0/tien/llm-compressor/data/generated_img'
    os.makedirs(img_save_dir, exist_ok=True)

    all_coco_promts_data = json.load(open(coco_val_dataset_path))
    all_coco_images = list(all_coco_promts_data.keys())

    pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16, cache_dir="/data0/tien/cache")
    pipe.to("cuda")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'device: {device}')
    
    print("Inserting activations quantizers ...")
    model = pipe.transformer

    all_modules = dict(model.named_modules())

    print("Calibrating")
    for name, module in all_modules.items():
        if isinstance(module , torch.nn.Linear) and 'transformer_blocks' in name:
            parent_module = all_modules['.'.join(name.split('.')[:-1])]
            quant_linear_layer = CollectInputsLinear(module, name)
            setattr(
                parent_module, name.split('.')[-1], quant_linear_layer
            )
    
    for i in range(0, 1*batch_size, batch_size): # len(all_coco_images)
        image_ids = all_coco_images[i:i+batch_size]
        input_prompts = [all_coco_promts_data[img] for img in image_ids]
        images = pipe(
            prompt=input_prompts,
            negative_prompt="",
            height=512,
            width=512,
            num_inference_steps=28,
            guidance_scale=7.0
        ).images

        for name, module in pipe.transformer.named_modules():
            if isinstance(module, CollectInputsLinear):
                module.forward_call = 0

    print("Quantizing")
    for name, module in all_modules.items():
        if isinstance(module, (CollectInputsLinear, torch.nn.Linear)) and 'transformer_blocks' in name:
            parent_module = all_modules['.'.join(name.split('.')[:-1])]
            quant_linear_layer = FP8Linear(module, name, quant_mode=quant_mode)
            setattr(
                parent_module, name.split('.')[-1], quant_linear_layer
            )

    print("Finish quant!")
    print(model)
    model.to(device)
    torch.set_grad_enabled(False)

    for i in range(0, 1*batch_size, batch_size): # len(all_coco_images)
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

        for j in range(len(image_ids)):
            images[j].save(os.path.join(img_save_dir, image_ids[j]))

        torch.cuda.empty_cache()
    print("Running time:", et-st)

quant()