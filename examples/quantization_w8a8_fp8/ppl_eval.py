import json
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

import pandas as pd

import time
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import tqdm
from custom_blocks import QuantLinear
from datasets import load_dataset


base_dir = os.path.join(os.path.dirname(__file__), '..', '..')
class Evaluator:
    def __init__(self, dataset, tokenizer, device, n_samples=None):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device

        self.dataset = tokenizer(
            '\n\n'.join(dataset['text']), return_tensors='pt'
        ).input_ids.to(device)

        self.n_samples = n_samples

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        nlls = []
        n_samples = self.n_samples if self.n_samples else self.dataset.size(1) // 2048
        run_time = []
        for i in tqdm.tqdm(range(n_samples), desc='Evaluating...'):
            batch = self.dataset[:, (i * 2048) : ((i + 1) * 2048)].to(model.device)
            
            st = time.time()
            with torch.no_grad():
                lm_logits = model(batch).logits
            et = time.time()
            run_time.append(et - st)

            shift_logits = lm_logits[:, :-1, :].contiguous().float()
            shift_labels = self.dataset[:, (i * 2048) : ((i + 1) * 2048)][:, 1:]
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            neg_log_likelihood = loss.float() * 2048
            nlls.append(neg_log_likelihood)

        return torch.exp(torch.stack(nlls).sum() / (n_samples * 2048)), run_time

full_results = {}
all_results = []
for model_path in [
    'meta-llama/Llama-3.2-1B',
    'meta-llama/Llama-3.2-1B-Instruct',
    'meta-llama/Llama-3.2-3B',
    'meta-llama/Llama-3.2-3B-Instruct',
    'meta-llama/Llama-3.1-8B',
    'meta-llama/Llama-3.1-8B-Instruct'
]:
    full_results[model_path] = {}
    for mode in [None, 'fp8', 'int8']:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        evaluator = Evaluator(dataset, tokenizer, 'cuda', n_samples=None)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, cache_dir=os.path.join(base_dir, 'data/cache')
        ).to('cuda')
        all_modules = dict(model.named_modules())
        if mode is not None:
            print('Quantizing')
            for name, module in all_modules.items():
                if isinstance(module, (torch.nn.Linear)):
                    parent_module = all_modules['.'.join(name.split('.')[:-1])]
                    quant_linear_layer = QuantLinear(module, name, quant_mode=mode)
                    setattr(
                        parent_module, name.split('.')[-1], quant_linear_layer
                    )
            print('Finish quantize')
        ppl, run_time = evaluator.evaluate(model)
        ppl = float(ppl.cpu().detach())

        rnt = sum(run_time[1:])/(len(run_time)-1)
        print(f'Mode: {mode}')
        print(f'Running time: {rnt}s')
        print(f'Perplexity: {ppl}')

        all_results.append([model_path, mode, ppl, rnt])

        if mode == None:
            mode = 'fp16'

        full_results[model_path][mode] = {
            'run_time': run_time,
            'ppl': ppl
        }

        del tokenizer, dataset, evaluator, model, all_modules
        torch.cuda.empty_cache()

        json.dump(full_results, open(os.path.join(base_dir, 'data/full_results_llm.json'), 'w'))

df = pd.DataFrame(all_results, columns = ['model', 'mode', 'ppl', 'batch_inference_time'])
df.to_csv(os.path.join(base_dir, 'data/full_results_df.csv'), index=False)
