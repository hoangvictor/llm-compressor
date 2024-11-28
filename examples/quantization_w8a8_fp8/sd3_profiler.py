# =========================================================================================================
# requirements.txt
# =========================================================================================================
"""
torch
accelerate
transformers
diffusers
sentencepiece
"""

import argparse
import time

import triton
import torch
import torch._dynamo.config
import torch._inductor.config
from diffusers import StableDiffusion3Pipeline

from torch.profiler import profile, record_function, ProfilerActivity

import re
import sys
import csv

torch.set_float32_matmul_precision("high")

torch._inductor.config.conv_1x1_as_mm = True
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.epilogue_fusion = False
torch._inductor.config.coordinate_descent_check_all_directions = True
torch._inductor.config.triton.unique_kernel_names = True
# Experimental features to reduce compilation times, will be on by default in future
torch._inductor.config.fx_graph_cache = True
torch._functorch.config.enable_autograd_cache = True

name_to_torch_types = {
    'fp16': torch.float16,
    'fp32': torch.float32,
}

def is_hip_available():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == 'hip'

def get_device_arch():
    target = triton.runtime.driver.active.get_current_target()
    arch = str(target.arch)
    if "gfx" in arch:
        return arch
    return "sm" + arch

def load_model(torch_dtype: torch.dtype, torch_compile: bool, profile: bool):
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch_dtype, cache_dir="/data0/tien/cache"
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    pipe.transformer.to(memory_format=torch.channels_last)
    pipe.vae.to(memory_format=torch.channels_last)
    
    compile_mode = 'max-autotune-no-cudagraphs'
    if torch_compile:
        pipe.transformer = torch.compile(pipe.transformer, fullgraph=True, mode=compile_mode)
        pipe.text_encoder = torch.compile(pipe.text_encoder, fullgraph=True, mode=compile_mode)
        pipe.text_encoder_2 = torch.compile(pipe.text_encoder_2, fullgraph=True, mode=compile_mode)
        pipe.text_encoder_3 = torch.compile(pipe.text_encoder_3, fullgraph=True, mode=compile_mode)
        pipe.vae.decode = torch.compile(pipe.vae.decode, fullgraph=True, mode=compile_mode)

    return pipe

# =========================================================================================================
# Benchmarking
# =========================================================================================================
def benchmark(pipe: StableDiffusion3Pipeline, prompt: str) -> StableDiffusion3Pipeline:
    import numpy as np
    device_arch = get_device_arch()
    for _ in range(10):
        _ = pipe(
            prompt=prompt,
            generator=torch.manual_seed(42),
            num_inference_steps=3,
            height=1024,
            width=1024,
            guidance_scale=7.0,
        )

    # Measure Performance
    records = []
    num_iters = 10
    for _ in range(num_iters):
        start = time.time()
        _ = pipe(
            prompt=prompt,
            generator=torch.manual_seed(42),
            num_inference_steps=50,
            height=1024,
            width=1024,
            guidance_scale=7.0,
        )
        records.append(time.time() - start)
    arr = np.array(records)
    _min = round(np.min(arr), 2)
    _max = round(np.max(arr), 2)
    p05 = round(np.percentile(arr,  5), 2)
    p25 = round(np.percentile(arr, 25), 2)
    p50 = round(np.percentile(arr, 50), 2)
    p75 = round(np.percentile(arr, 75), 2)
    p95 = round(np.percentile(arr, 95), 2)

    print(f"Time to run SD3 on {device_arch}")
    print(f"====================================================")
    print(f"Min\tP05\tP25\tP50\tP75\tP95\tMax")
    print(f"{_min}\t{p05}\t{p25}\t{p50}\t{p75}\t{p95}\t{_max}")


# =========================================================================================================
# Profiling
# =========================================================================================================
def profile(pipe: StableDiffusion3Pipeline, prompt: str, profiler_output: str) -> None:
    # Warmup
    _ = pipe(
        prompt=prompt,
        generator=torch.manual_seed(42),
        num_inference_steps=1,
        height=1024,
        width=1024,
        guidance_scale=7.0,
    )

    with torch.profiler.profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=True
        ) as prof:
        _ = pipe(
            prompt=prompt,
            generator=torch.manual_seed(42),
            num_inference_steps=50,
            height=1024,
            width=1024,
            guidance_scale=7.0,
        )

    print(f"Writing to {profiler_output}.profile .....")
    with open(f"{profiler_output}.profile", 'w') as f:
        table = prof.key_averages(
                group_by_input_shape=True
        ).table(
                sort_by="cuda_time_total", 
                row_limit=10000,
                max_name_column_width=10000
        )
        f.write(table)

    print(f"Converting .profile to .csv ......")
    with open(f"{profiler_output}.profile", 'r') as f:
        lines = f.readlines()
    data = []
    for line in lines:
        columns = re.split(r'\s{2,}', line.strip())
        data.append(columns)
    output_file = f"{profiler_output}.csv"
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        for row in data:
            writer.writerow(row)

    print(f"Writing to {profiler_output}.json .....")
    prof.export_chrome_trace(f"{profiler_output}.json")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)

    default_prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
    parser.add_argument("--prompt", type=str, default=default_prompt)
    parser.add_argument("--dtype", type=str, default='fp16', help="Model data type")
    parser.add_argument("--profile", action='store_true', default=False)
    parser.add_argument("--benchmark", action='store_true', default=False)
    parser.add_argument("--compile", action='store_true', default=False)
    parser.add_argument("--outdir", type=str, default='./output', help="Output directory")
    args = parser.parse_args()

    assert args.profile or args.benchmark, "Please specify `--profile` or `--benchmark` mode"

    dtype = args.dtype
    use_torch_compile = args.compile
    sd3_pipe = load_model(name_to_torch_types[dtype], use_torch_compile, args.profile)

    prompt = args.prompt
    if args.profile:
        print(f"Profiling SD3 on {dtype} with torch.compile={use_torch_compile}")
        device_arch = get_device_arch()
        compile_str = "torch_compile" if use_torch_compile else "vanilla"
        file_output = f"{args.outdir}/sd3_{dtype}_{compile_str}_{device_arch}"
        profile(pipe=sd3_pipe, prompt=prompt, profiler_output=file_output)

    if args.benchmark:
        print(f"Benchmarking SD3 on {dtype} with torch.compile={use_torch_compile}")
        benchmark(pipe=sd3_pipe, prompt=prompt)
