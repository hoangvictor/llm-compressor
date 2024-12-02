import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

from typing import Union

import torch
import torch._dynamo.config
import torch._inductor.config

from torch.nn import Linear
from custom_blocks import QuantLinear

from torch.profiler import ProfilerActivity
from transformers import AutoModelForCausalLM

import re
import csv

def profile(linear_layer: Union[Linear, QuantLinear], profiler_output: str) -> None:
    # Warmup
    input_shape = (linear_layer.in_features, linear_layer.weight.shape[1])
    input = torch.rand(input_shape).to(torch.float16).to('cuda')

    # Warmup
    _ = linear_layer(input)

    with torch.profiler.profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=True
        ) as prof:
        _ = linear_layer(input)

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

model_path = "meta-llama/Llama-3.2-3B"
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.float16, cache_dir='/llm-compressor-workspace/data/cache'
).to('cuda')
all_modules = dict(model.named_modules())

for name, module in all_modules.items():
    if isinstance(module, (torch.nn.Linear)):
        break
fp8_module = QuantLinear(module, name, 'fp8')
int8_module = QuantLinear(module, name, 'int8')

profile(module, 'original')
profile(fp8_module, 'fp8')
profile(int8_module, 'int8')
