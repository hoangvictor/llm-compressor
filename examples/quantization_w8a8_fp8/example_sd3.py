import torch
import json

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import torch
import torch.nn as nn
from diffusers import StableDiffusion3Pipeline
from vllm.model_executor.layers.quantization.utils.w8a8_utils import apply_fp8_linear, apply_int8_linear

base_dir = os.path.join(os.path.dirname(__file__), '..', '..')
quant_mode = 'int8'
save_images = True
activations_scales = {}
quant_errors = {}
force_run = True

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

    @torch.no_grad()
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if len(input.shape) == 2:
            current_max = torch.max(torch.abs(input)).reshape(1,)
        else:
            current_max = torch.max(torch.max(torch.abs(input.reshape(input.shape[0], -1, input.shape[-1])), dim=0)[0], dim=1)[0]
        if self.old_module_name not in activations_scales:
            activations_scales[self.old_module_name] = current_max
        else:
            activations_scales[self.old_module_name] = torch.max(
                current_max, 
                activations_scales[self.old_module_name]
            )
        self.forward_call += 1
        return super().forward(input)

    def extra_repr(self) -> str:
        return f"in_features={self.fp8_weight.shape[0]}, out_features={self.fp8_weight.shape[1]}, bias={self.bias is not None}, quant_mode={self.quant_mode}"


class FP8Linear(nn.Module):

    def __init__(
        self,
        module: nn.Linear,
        old_module_name: str = None,
        quant_mode: str = 'fp8',
        collect_quant_error: bool = False
    ):
        super().__init__()        
        self.local_scales = []

        max_abs_val = torch.max(torch.abs(module.weight.T), axis=0).values.to(torch.float32)

        self.fp8_data_type = torch.float8_e4m3fn
        if torch.version.hip is not None:
            self.fp8_data_type = torch.float8_e4m3fnuz

        finfo = torch.finfo(self.fp8_data_type)
        iinfo = torch.iinfo(torch.int8)
        
        assert quant_mode in ['fp8', 'int8', None]

        self.quant_mode = quant_mode
        self.old_module_name = old_module_name
        self.collect_quant_error = collect_quant_error
        self.original_weight = module.weight
        
        self.input_scale = activations_scales[self.old_module_name].to(torch.float32) # None
        if quant_mode == 'fp8':
            self.input_scale /= finfo.max
        else:
            self.input_scale /= iinfo.max
        self.fp8_weight_scale = (max_abs_val /  finfo.max).clamp(min=1e-12)
        self.fp8_weight_scale = self.fp8_weight_scale.contiguous()
        self.fp8_weight = (module.weight.T / self.fp8_weight_scale).clamp(min=finfo.min, max= finfo.max)

        self.fp8_weight = self.fp8_weight.to(self.fp8_data_type)
        
        self.int8_weight_scale = (max_abs_val / iinfo.max).clamp(min=1e-12)
        self.int8_weight_scale = self.int8_weight_scale.contiguous()
        self.int8_weight = (module.weight.T / self.int8_weight_scale).clamp(min=iinfo.min, max=iinfo.max).to(torch.int8)

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
            output = apply_fp8_linear(input_2d, self.fp8_weight, input_scale=scales, weight_scale=self.fp8_weight_scale, bias=self.bias, use_per_token_if_dynamic=True)
            output = output.view(*output_shape)
        elif self.quant_mode == 'int8':
            output = apply_int8_linear(input_2d, self.int8_weight, input_scale=scales, weight_scale=self.int8_weight_scale, bias=self.bias)
            output = output.view(*output_shape)
        elif self.quant_mode is None:
            output = nn.functional.linear(input, self.original_weight, self.bias)
        else:
            raise Exception(f"The forward pass of this quant mode `{self.quant_mode}` is not implemented yet.")

        if self.collect_quant_error:
            original_output = nn.functional.linear(input, self.original_weight, self.bias)
            current_error = torch.mean(torch.abs(original_output - output))
            if self.old_module_name not in quant_errors:
                quant_errors[self.old_module_name] = [current_error]
            else:
                quant_errors[self.old_module_name].append(current_error)
        return output


    def extra_repr(self) -> str:
        return f"in_features={self.fp8_weight.shape[0]}, out_features={self.fp8_weight.shape[1]}, bias={self.bias is not None}, quant_mode={self.quant_mode}, collect_quant_error={self.collect_quant_error}"


def quant():
    global activations_scales, quant_errors

    batch_size = 32
    num_batch = 1
    coco_val_dataset_path = f'{base_dir}/data/coco_all_data_info.json'
    img_save_dir = f'{base_dir}/data/generated_img'
    os.makedirs(img_save_dir, exist_ok=True)

    all_coco_promts_data = json.load(open(coco_val_dataset_path))
    all_coco_images = list(all_coco_promts_data.keys())

    pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16, cache_dir=os.path.join(base_dir, 'data/cache'))
    pipe.to("cuda")
    model = pipe.transformer
    all_modules = dict(model.named_modules())

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'device: {device}')
    
    print("Inserting activations quantizers ...")

    if not os.path.exists(f'{base_dir}/data/activations_scales.pt'):
        print("Calibrating")
        for name, module in all_modules.items():
            if isinstance(module , torch.nn.Linear) and 'transformer_blocks' in name:
                parent_module = all_modules['.'.join(name.split('.')[:-1])]
                quant_linear_layer = CollectInputsLinear(module, name)
                setattr(
                    parent_module, name.split('.')[-1], quant_linear_layer
                )
        
        for i in range(0, 1024, batch_size): # len(all_coco_images)
            image_ids = all_coco_images[i:i+batch_size]
            input_prompts = [all_coco_promts_data[img] for img in image_ids]
            _ = pipe(
                prompt=input_prompts,
                negative_prompt="",
                height=512,
                width=512,
                num_inference_steps=28,
                guidance_scale=7.0
            )

            for name, module in pipe.transformer.named_modules():
                if isinstance(module, CollectInputsLinear):
                    module.forward_call = 0
        
        torch.save(activations_scales, f'{base_dir}/data/activations_scales.pt')
    else:
        activations_scales = torch.load(f'{base_dir}/data/activations_scales.pt')

    print("Quantizing")
    for name, module in all_modules.items():
        if isinstance(module, (CollectInputsLinear, torch.nn.Linear)) and 'transformer_blocks' in name:
            parent_module = all_modules['.'.join(name.split('.')[:-1])]
            quant_linear_layer = FP8Linear(module, name, quant_mode=None)
            setattr(
                parent_module, name.split('.')[-1], quant_linear_layer
            )

    print("Finish quant!")
    print(model)
    model.to(device)
    torch.set_grad_enabled(False)

    for quant_mode in ['int8', 'fp8']:
        if force_run or not os.path.exists(f'{base_dir}/data/quant_errors_{quant_mode}.pt'):
            print("Collecting Quantization Error")
            all_modules = dict(model.named_modules())
            for current_module_name, current_module in all_modules.items():
                if not isinstance(current_module, (FP8Linear)):
                    continue
                print(f"Current module: {current_module_name}")
                current_module.collect_quant_error = True
                current_module.quant_mode = quant_mode
                for other_module_name, other_module in all_modules.items():
                    if other_module_name == current_module_name:
                        continue
                    if not isinstance(other_module, (FP8Linear)):
                        continue
                    other_module.collect_quant_error = False
                    other_module.quant_mode = None
                
                for i in range(0, num_batch*batch_size, batch_size):
                    image_ids = all_coco_images[i:i+batch_size]
                    input_prompts = [all_coco_promts_data[img] for img in image_ids]
                    _ = pipe(
                        prompt=input_prompts,
                        negative_prompt="",
                        height=512,
                        width=512,
                        num_inference_steps=28,
                        guidance_scale=7.0
                    )
                    torch.cuda.empty_cache()
            for layer in quant_errors:
                quant_errors[layer] = torch.mean(torch.Tensor(quant_errors[layer]))
            torch.save(quant_errors, f'{base_dir}/data/quant_errors_{quant_mode}.pt')
        else:
            quant_errors = torch.load(f'{base_dir}/data/quant_errors_{quant_mode}.pt')

quant()