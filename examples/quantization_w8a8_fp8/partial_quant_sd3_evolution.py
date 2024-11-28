import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

import glob
import torch
import json
import random

import numpy as np

import torch
import torch.nn.functional as F
from diffusers import StableDiffusion3Pipeline

from PIL import Image
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.multimodal.clip_score import CLIPScore

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

def load_images(image_path, img_width=512, img_height=512):
    def preprocess_image(image, img_width=512, img_height=512):
        image = torch.tensor(image).unsqueeze(0)
        image = image.permute(0, 3, 1, 2)
        return F.center_crop(image, (img_width, img_height))

    images_lst = [
        np.array(Image.open(path).convert("RGB"))
        for path in image_path
    ]
    images_lst = torch.cat([
        preprocess_image(image, img_width, img_height) for image in images_lst
    ])
    return images_lst

@torch.no_grad()
def comp_fid(
    real_files_dir,
    fake_files_dir,
    batch_size=32,
    width=512,
    height=512
):
    total = len(fake_files_dir)
    num = max(total // batch_size, 1)
    fid = FrechetInceptionDistance(normalize=True).to('cuda')
    
    fake_files = glob.glob(os.path.join(fake_files_dir, '*.jpg'))
    fake_imgs = [f.split('/')[-1] for f in fake_files]
    real_files = [os.path.join(real_files_dir, f) for f in fake_imgs]
    
    for i in range(num):
        real_images = load_images(fake_files[batch_size * i:batch_size *
                                             (i + 1)],
                                  img_width=width,
                                  img_height=height)
        fid.update(real_images.to('cuda'), real=True)

    for i in range(num):
        fake_images = load_images(real_files[batch_size * i:batch_size *
                                                 (i + 1)],
                                  img_width=width,
                                  img_height=height)
        fid.update(fake_images.to('cuda'), real=False)
    fid_score = float(fid.compute())
    return fid_score

class EvolutionSearcher:
    def __init__(
        self,
        pipe,
        all_coco_images,
        all_coco_promts_data,
        real_img_dir: str,
        img_save_dir: str = None,
        population_num: int = 50,
        max_epochs: int = 10,
        select_num: int = 10,
        mutation_num: int = 5,
        crossover_num: int = 5,
        m_prob: float = 0.1,
        num_batch: int = 1,
        batch_size: int = 32
    ):  
        self.epoch = 0
        self.population_num = population_num
        self.select_num = select_num
        self.max_epochs = max_epochs
        self.mutation_num = mutation_num
        self.crossover_num = crossover_num
        self.m_prob = m_prob
        self.quant_modes = ['fp8', 'int8']
        self.real_img_dir = real_img_dir
        
        if img_save_dir is None:
            img_save_dir = f'{base_dir}/data/generated_img'
        self.img_save_dir = img_save_dir
        os.makedirs(self.img_save_dir, exist_ok=True)

        self.candidates = []
        self.candidates_loss = {}

        self.all_coco_images = all_coco_images
        self.all_coco_promts_data = all_coco_promts_data
        self.img_save_dir = img_save_dir

        self.pipe = pipe
        self.model = pipe.transformer
        self.model.to('cuda')

        self.num_batch = num_batch
        self.batch_size = batch_size

        self.all_modules = dict(self.model.named_modules())
        self.linear_layers = [name for name, module in self.all_modules.items() if isinstance(module, torch.nn.Linear) and 'transformer_blocks' in name]
        for quant_mode in self.quant_modes:
            os.makedirs(f'{self.img_save_dir}_{quant_mode}', exist_ok=True)

    def sample_subnet(self):
        subnet = random.choices(self.quant_modes, k=len(self.linear_layers))
        return subnet

    def get_cand_loss(self, cand):
        mi = 0
        cand = eval(cand)
        for name, module in self.all_modules.items():
            if isinstance(module, (torch.nn.Linear, QuantLinear)) and 'transformer_blocks' in name:
                parent_module = self.all_modules['.'.join(name.split('.')[:-1])]
                quant_linear_layer = QuantLinear(module, name, quant_mode=cand[mi])
                setattr(
                    parent_module, name.split('.')[-1], quant_linear_layer
                )
                mi += 1

        for i in range(0, self.num_batch*self.batch_size, self.batch_size):
            image_ids = self.all_coco_images[i:i+self.batch_size]
            input_prompts = [self.all_coco_promts_data[img] for img in image_ids]

            images = self.pipe(
                prompt=input_prompts,
                negative_prompt="",
                height=512,
                width=512,
                num_inference_steps=28,
                guidance_scale=7.0
            ).images

            for j in range(len(image_ids)):
                images[j].save(os.path.join(self.img_save_dir, image_ids[j]))
            
            torch.cuda.empty_cache()

        self.candidates_loss[cand] = comp_fid(
            real_files_dir=self.real_img_dir,
            fake_files_dir=self.img_save_dir,
            batch_size=self.batch_size
        )

    def get_cross(self):
        res = []
        max_iters = self.cross_num * 10

        def random_cross():
            cand1 = random.choice(self.top_candidates[:self.select_num])
            cand2 = random.choice(self.top_candidates[:self.select_num])

            new_cand = []
            cand1 = eval(cand1)
            cand2 = eval(cand2)

            for i in range(len(cand1)):
                if np.random.random_sample() < 0.5:
                    new_cand.append(cand1[i])
                else:
                    new_cand.append(cand2[i])
            return new_cand

        while len(res) < self.cross_num and max_iters > 0:
            max_iters -= 1
            cand = random_cross()
            cand = str(cand)
            if cand in self.candidates_loss:
                continue
            res.append(cand)
            print('cross {}/{}'.format(len(res), self.cross_num))

        print('cross_num = {}'.format(len(res)))
        return res

    def get_mutation(self):
        res = []
        max_iters = self.mutation_num * 10

        def random_func():
            cand = random.choice(self.top_candidates[:self.select_num])
            cand = eval(cand)
            for i in range(len(cand)):
                if random.random() < self.m_prob:
                    cand[i] = random.choice(self.quant_modes)
            return cand

        while len(res) < self.mutation_num and max_iters > 0:
            max_iters -= 1
            cand = random_func()
            cand = str(cand)
            if cand in self.candidates_loss:
                continue
            res.append(cand)
            print('mutation {}/{}'.format(len(res), self.mutation_num))

        print('mutation_num = {}'.format(len(res)))
        return res

    def get_random(self):
        max_iters = self.population_num * 10

        while len(self.candidates) < self.population_num and max_iters > 0:
            max_iters -= 1
            cand = self.sample_subnet()
            cand = str(cand)
            if cand in self.candidates_loss:
                continue
            self.candidates.append(cand)
        
        for cand in self.candidates:
            self.candidates_loss[cand] = self.get_cand_loss(cand)

    def update_top_k(self):
        self.top_candidates = sorted(self.candidates, key=lambda cand: self.candidates_loss[cand])

    def search(self):
        preset = []
        preset.append([128]*112)
        self.get_random()

        while self.epoch < self.max_epochs:
            print('epoch = {}'.format(self.epoch))
            self.update_top_k(self.candidates)
            print('epoch = {} : top {} result'.format(
                self.epoch, self.select_num))
            for i, cand in enumerate(self.select_num):
                print('No.{} {} loss = {}'.format(
                    i + 1, cand, self.candidates_loss[cand]))
            
            mutation = self.get_mutation()

            self.candidates = mutation

            cross_cand = self.get_cross()
            self.candidates += cross_cand

            self.get_random()

            self.epoch += 1

def evolution_search():
    coco_val_dataset_path = f'{base_dir}/data/coco_all_data_info.json'
    real_img_dir = f'{base_dir}/data/real_img'

    all_coco_promts_data = json.load(open(coco_val_dataset_path))
    all_coco_images = list(all_coco_promts_data.keys())

    pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16, cache_dir="/data0/tien/cache")
    pipe.to("cuda")
    pipe.transformer.to(memory_format=torch.channels_last)
    pipe.vae.to(memory_format=torch.channels_last)

    evolution_searcher = EvolutionSearcher(
        pipe=pipe,
        all_coco_images=all_coco_images,
        all_coco_promts_data=all_coco_promts_data,
        real_img_dir=real_img_dir
    )
    evolution_searcher.search()

evolution_search()