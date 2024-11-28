import argparse
from copy import deepcopy
import glob
import json
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from diffusers import DiffusionPipeline, StableDiffusionXLPipeline
from PIL import Image
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.multimodal.clip_score import CLIPScore
from torchvision.transforms import functional as F
from tqdm.auto import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset


class Model_Handle():

    def __init__(self, model_type, dtype):
        self.model_type = model_type  # Initialize model as None, will be loaded later
        self.dtype = dtype

    def _load_from_huggingface(self, model_name: str):
        """
        Load model from Hugging Face model card.
        Args:
            model_name (str): The name of the model from Hugging Face to load.
        """
        print(f"Loading model '{model_name}' from Hugging Face...")
        pipe = DiffusionPipeline.from_pretrained(model_name,
                                                 torch_dtype=self.dtype,
                                                 use_safetensors=True,
                                                 variant="fp16")
        return pipe

    def _load_from_checkpoint(self, checkpoint_path: str):
        """
        Load model from a local checkpoint.
        Args:
            checkpoint_path (str): Path to the .ckpt file to load.
        """
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint '{checkpoint_path}' does not exist.")

        print(f"Loading model from checkpoint '{checkpoint_path}'...")
        checkpoint = torch.load(checkpoint_path,
                                map_location=torch.device('cpu'))
        model = nn.Module()
        model.load_state_dict(checkpoint['model_state_dict']).to(self.dtype)
        return model

    def load_model(self, model_name):
        if self.model_type == 'ckpt':
            return self._load_from_checkpoint(model_name)
        elif self.model_type == 'hf':
            return self._load_from_huggingface(model_name)
        else:
            return None

class ImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")  # Open image and convert to RGB
        if self.transform:
            image = self.transform(image)  # Apply transformation if provided
        return image, img_path  # Return image and its path for reference

def parse_args():
    parser = argparse.ArgumentParser(
        description="Simple example of a inference script.")

    parser.add_argument(
        "--start",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--end",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--gen_img",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--compute_fid",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--compute_clip",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--count_missmatch",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--sub_ds_value",
        type=int,
        default=None,
        help=
        "Value to create a sub-dataset. If not specified, the whole dataset will be used."
    )
    parser.add_argument("--images_path",
                        type=str,
                        default="/home/tien/project/Q-DiT/data/val2014/",
                        help="Path to the image directory")
    parser.add_argument("--captions_path",
                        default="/home/tien/project/Q-DiT/data/captions_val2014.json",
                        type=str,
                        help="Path to the COCO caption json file")
    parser.add_argument('--gen_bs',
                        type=int,
                        default=64,
                        help="Batch size for image generation")
    parser.add_argument("--gen_img_dir",
                        type=str,
                        default='/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.0_samples_64/',
                        help="Path to generated images directory")
    parser.add_argument("--width",
                        type=int,
                        default=512,
                        help="generated image width")
    parser.add_argument("--height",
                        type=int,
                        default=512,
                        help="generated image height")
    parser.add_argument("--infer_steps",
                        type=int,
                        default=28,
                        help="inference steps for model")
    parser.add_argument("--guidance_scale",
                        type=float,
                        default=7.0,
                        help="guidance scale for free guidance")
    parser.add_argument('--dtype',
                        choices=['fp16', 'fp32'],
                        default='fp16',
                        help="Data type for model inference")
    args = parser.parse_args()
    return args


def get_captions_imgs(file_path):
    """
    Get dictionary of image and caption with key is image id and 
    value is another dictionary of images and their captions
    Args:
        file_path : path to json file

    Returns:
        ds_dict
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
    ds_dict = dict()
    for i in data['images']:
        ds_dict[i['id']] = {'file_name': i['file_name'], 'captions': []}
    for item in data['annotations']:
        ds_dict[item['image_id']]['captions'].append(item['caption'])
    return ds_dict


def count_missmatch(gen_img_paths, caption_img_pair_dict, verbose=False):
    """
    Compare the images in original dataset and generated dataset
    Args:
        gen_img_paths: path to real image dataset
        caption_img_pair_dict: dictionary of caption and image
        verbose (bool, optional)

    Returns:
        True if all the image matches, else False
    """
    total = len(caption_img_pair_dict)
    total_matching = 0
    for i in gen_img_paths:
        img_path = str(i.split('.')[0] + '.jpg').split('/')[-1]
        img_path = img_path.split('/')[-1]
        try:
            total_matching += min(len(caption_img_pair_dict[img_path]), 1)
        except Exception as e:
            print(e)
    if verbose:
        print(f'total matching: {total_matching}/{total}')
    if total_matching % total == 0:
        return True
    else:
        return False

def preprocess_image(image, img_width=512, img_height=512):
    """
    Preprocess images for computing FID score.
    Args:
        image (PIL.Image)
        img_size (int, optional): Image size. Defaults to 512.

    Returns:
        (PIL.Image): preprocessed image
    """
    image = torch.tensor(image).unsqueeze(0)
    image = image.permute(0, 3, 1, 2)
    return F.center_crop(image, (img_width, img_height))


def load_images(image_path, img_dir=None, img_width=512, img_height=512):
    """
    Load images and preprocess in batche.
    Args:
        image_path (Path): List of image path
        batch_size (int, optional): Batch size. Defaults to 32.

    Returns:
        List: List of preprocessed images.
    """
    if img_dir is None:
        img_dir = ''
    images_lst = [
        np.array(Image.open(img_dir + path).convert("RGB"))
        for path in image_path
    ]
    images_lst = torch.cat([
        preprocess_image(image, img_width, img_height) for image in images_lst
    ])
    return images_lst


def get_gen_image_path(gen_image_dir):
    fake_image_path = []
    for ext in ('*.jpg', '*.png'):
        fake_image_path.extend(glob.glob(os.path.join(gen_image_dir, ext)))
    return fake_image_path


def gen_img(pipeline,
            prompts,
            file_names,
            batch_size,
            output_dir,
            width=512,
            height=512,
            guidance_scale=7.5,
            infer_steps=50,
            img_format='.jpg'):
    """
    Generate images using diffusers SD pipeline. Saving the generated image with the same name as the original image.
    Args:
        pipeline (_type_): diffusers StableDiffusion pipeline
        prompts (str): list of prompt to generate images
        file_names (str): list of image paths use to get the image format and save.
        batch_size (int): batch size
        output_dir (str): path to the directory to save the generated images
        width (int, optional): width. Defaults to 512.
        height (int, optional): height. Defaults to 512.
        guidance_scale (float, optional): guidance_scale. Defaults to 7.5.
        infer_steps (int, optional): infer_steps. Defaults to 50.
        img_format (str, optional): img format for generated image. Defaults to .jpg.
    """
    total = len(prompts)
    num = total // batch_size

    for i in tqdm(range(num),
                  desc='Generating image samples for FID evaluation.'):
        print(f"i/num: {i}/{num}")
        fake_images = pipeline(
            prompt=prompts[batch_size * i:batch_size * (i + 1)],
            num_inference_steps=infer_steps,
            output_type="pil",
            height=height,
            width=width,
            guidance_scale=guidance_scale,
        ).images
        for img, file_name in zip(
                fake_images, file_names[batch_size * i:batch_size * (i + 1)]):
            file_name = str(file_name).split('.')[0]
            img.save(f'{output_dir+file_name}.{img_format}')


def model_predict(model, data_loader, gen_img_dir, device, batch_size=60):
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for batch_idx, (batch_images, batch_image_paths) in enumerate(data_loader):
            batch_images = batch_images.to(device)
            outputs = model(batch_images)
            
            # Loop through each image in the batch
            for i, (output, img_path) in enumerate(zip(outputs, batch_image_paths)):
                image_tensor = output.cpu()
                image = F.to_pil_image(image_tensor)

                # Save the PIL image with a new filename
                save_path = os.path.join(gen_img_dir, f"processed_{batch_idx * batch_size + i}.jpg")
                image.save(save_path)

                print(f"Saved {save_path}")


@torch.no_grad()
def comp_fid(fid,
             file_names,
             gen_image_path,
             device,
             img_dir=None,
             batch_size=64,
             width=512,
             height=512):
    """
    Compute FID score by loading real images dataset and generated images
    Args:
        fid : Frechet Inception Distance
        image_path: path to real images dataset
        gen_image_path : path to generated images
        device: CPU or GPU
        batch_size (int, optional): Batch size to update FID. Defaults to 64.

    Returns:
        fid_scores: FID score
    """
    print('Compute FID:')
    total = len(file_names)
    num = total // batch_size
    fid = FrechetInceptionDistance(normalize=True).to(device)
    
    print("Load real images")
    for i in tqdm(range(num)):
        real_images = load_images(file_names[batch_size * i:batch_size *
                                             (i + 1)],
                                  img_dir=img_dir,
                                  img_width=width,
                                  img_height=height)
        fid.update(real_images.to(device), real=True)

    print("Load fake images")
    for i in tqdm(range(num)):
        fake_images = load_images(gen_image_path[batch_size * i:batch_size *
                                                 (i + 1)],
                                  img_width=width,
                                  img_height=height)
        fid.update(fake_images.to(device), real=False)

    fid_score = float(fid.compute())
    return fid_score

@torch.no_grad()
def comp_clip(caption_img_pair_dict, fake_image_path, device, batch_size=150):
    """
    Compute CLIP score using openai/clip-vit-base-patch3
    Args:
        caption_img_pair_dict (str): a dictionary with image_file is key and caption is value
        fake_image_path : path to the generated images
        device : GPU or CPU
        batch_size: batch size to load images to update Defaults to 150.

    Returns:
        clip_scores: CLIP score
    """
    print('Compute CLIP:')
    metric = CLIPScore(
        model_name_or_path="openai/clip-vit-base-patch32").to(device)
    clip_scores = []
    img_names = [img_path.split('/')[-1].split('.')[0] for img_path in fake_image_path]
    caption_img_pair_dict = {k:caption_img_pair_dict[k] for k in img_names}
    total = len(caption_img_pair_dict.keys())
    num_batches = total // batch_size + (1 if total % batch_size != 0 else 0)
    for i in tqdm(range(num_batches),
                  desc='Compute CLIP score for gen images'):
        batch_imgs = fake_image_path[batch_size * i:batch_size * (i + 1)]
        flat_prompts = []
        for img_path in batch_imgs:
            img_path = img_path.split('/')[-1]
            flat_prompts.append(caption_img_pair_dict[img_path.split('.')[0]][0])
        # Flatten the list of prompts
        images = [
            np.array(Image.open(str(file_name)).convert("RGB"))
            for file_name in batch_imgs
        ]
        images = torch.cat([preprocess_image(image) for image in images])

        metric.update(images=images.to(device), text=flat_prompts)

    clip_scores = float(metric.compute())
    clip_scores = np.array(clip_scores)
    return np.mean(clip_scores) / 100

@torch.no_grad()
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')

    width, height = args.width, args.height
    ds_dict = get_captions_imgs(args.captions_path)
    img_format = str(list(ds_dict.values())[0]['file_name']).split('.')[-1]
    caption_img_pair_dict = {
        v['file_name'].split('.')[0]: v['captions']
        for v in list(ds_dict.values())[:args.sub_ds_value]
    }
    captions, file_names = list(caption_img_pair_dict.values()), list(
        caption_img_pair_dict.keys())
    captions = [c[0] for c in captions]
    file_names = [f+'.'+img_format for f in file_names]
    original_file_names = deepcopy(file_names)

    start = args.start
    end = args.end
    
    batch_size = args.batch_size
    results = []
    all_dirs = ['/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.0_samples_1024', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.0_samples_1152', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.0_samples_128', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.0_samples_1280', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.0_samples_1408', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.0_samples_1536', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.0_samples_1664', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.0_samples_1792', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.0_samples_1920', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.0_samples_2048', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.0_samples_256', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.0_samples_384', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.0_samples_512', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.0_samples_64', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.0_samples_640', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.0_samples_768', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.0_samples_896', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.1', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.1_samples_1024', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.1_samples_1152', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.1_samples_128', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.1_samples_1280', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.1_samples_1408', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.1_samples_1536', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.1_samples_1664', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.1_samples_1792', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.1_samples_1920', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.1_samples_2048', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.1_samples_256', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.1_samples_384', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.1_samples_512', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.1_samples_64', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.1_samples_640', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.1_samples_768', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.1_samples_896', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.2', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.25', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.2_samples_1024', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.2_samples_1152', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.2_samples_128', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.2_samples_1280', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.2_samples_1408', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.2_samples_1536', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.2_samples_1664', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.2_samples_1792', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.2_samples_1920', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.2_samples_2048', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.2_samples_256', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.2_samples_384', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.2_samples_512', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.2_samples_64', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.2_samples_640', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.2_samples_768', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.2_samples_896', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.3', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.3_samples_1024', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.3_samples_1152', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.3_samples_128', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.3_samples_1280', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.3_samples_1408', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.3_samples_1536', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.3_samples_1664', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.3_samples_1792', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.3_samples_1920', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.3_samples_2048', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.3_samples_256', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.3_samples_384', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.3_samples_512', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.3_samples_64', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.3_samples_640', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.3_samples_768', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.3_samples_896', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.4', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.4_samples_1024', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.4_samples_1152', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.4_samples_128', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.4_samples_1280', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.4_samples_1408', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.4_samples_1536', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.4_samples_1664', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.4_samples_1792', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.4_samples_1920', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.4_samples_2048', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.4_samples_256', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.4_samples_384', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.4_samples_512', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.4_samples_64', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.4_samples_640', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.4_samples_768', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.4_samples_896', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.5', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.5_samples_1024', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.5_samples_1152', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.5_samples_128', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.5_samples_1280', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.5_samples_1408', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.5_samples_1536', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.5_samples_1664', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.5_samples_1792', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.5_samples_1920', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.5_samples_2048', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.5_samples_256', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.5_samples_384', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.5_samples_512', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.5_samples_64', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.5_samples_640', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.5_samples_768', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.5_samples_896', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.6', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.6_samples_1024', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.6_samples_1152', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.6_samples_128', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.6_samples_1280', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.6_samples_1408', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.6_samples_1536', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.6_samples_1664', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.6_samples_1792', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.6_samples_1920', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.6_samples_2048', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.6_samples_256', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.6_samples_384', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.6_samples_512', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.6_samples_64', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.6_samples_640', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.6_samples_768', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.6_samples_896', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.7', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.75', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.7_samples_1024', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.7_samples_1152', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.7_samples_128', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.7_samples_1280', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.7_samples_1408', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.7_samples_1536', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.7_samples_1664', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.7_samples_1792', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.7_samples_1920', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.7_samples_2048', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.7_samples_256', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.7_samples_384', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.7_samples_512', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.7_samples_64', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.7_samples_640', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.7_samples_768', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.7_samples_896', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.8', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.8_samples_1024', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.8_samples_1152', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.8_samples_128', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.8_samples_1280', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.8_samples_1408', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.8_samples_1536', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.8_samples_1664', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.8_samples_1792', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.8_samples_1920', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.8_samples_2048', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.8_samples_256', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.8_samples_384', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.8_samples_512', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.8_samples_64', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.8_samples_640', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.8_samples_768', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.8_samples_896', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.9', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.9_samples_1024', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.9_samples_1152', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.9_samples_128', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.9_samples_1280', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.9_samples_1408', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.9_samples_1536', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.9_samples_1664', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.9_samples_1792', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.9_samples_1920', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.9_samples_2048', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.9_samples_256', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.9_samples_384', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.9_samples_512', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.9_samples_64', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.9_samples_640', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.9_samples_768', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_0.9_samples_896', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_1.0_samples_1024', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_1.0_samples_1152', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_1.0_samples_128', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_1.0_samples_1280', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_1.0_samples_1408', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_1.0_samples_1536', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_1.0_samples_1664', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_1.0_samples_1792', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_1.0_samples_1920', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_1.0_samples_2048', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_1.0_samples_256', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_1.0_samples_384', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_1.0_samples_512', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_1.0_samples_64', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_1.0_samples_640', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_1.0_samples_768', '/home/tien/project/Q-DiT/data/full_coco_w6a6_rate_1.0_samples_896']
    if end == -1:
        end = len(all_dirs)
    for gen_img_dir in all_dirs[start:end]:
        gen_img_dir_name = gen_img_dir.split('/')[-1]
        if not '_rate' in gen_img_dir_name:
            continue
        if os.path.exists(f'/home/tien/project/Q-DiT/data/results_{gen_img_dir_name}.csv'):
            continue
        
        print(gen_img_dir_name)
        gen_img_path = get_gen_image_path(gen_img_dir)

        gen_img_path = [gen_img for gen_img in gen_img_path if gen_img.split('/')[-1] in original_file_names]
        gen_img_file_names = set(gen_img.split('/')[-1] for gen_img in gen_img_path)
        file_names = [fn for fn in original_file_names if fn in gen_img_file_names]

        assert set(file_names) == set(gen_img_file_names)
        if 'samples' in gen_img_dir:
            assert len(set(file_names)) == int(gen_img_dir_name.split('_')[-1])
        else:
            assert len(set(file_names)) == len(captions)
        
        if args.compute_fid:
            fid = FrechetInceptionDistance(normalize=True).to(device)
            fid_score = comp_fid(
                fid,
                file_names,
                gen_img_path,
                img_dir=args.images_path,
                device=device,
                batch_size=batch_size,
                width=width,
                height=height,
            )
            print(f"FID: {fid_score:.3f}")

        if args.compute_clip:
            clip_score = comp_clip(caption_img_pair_dict, gen_img_path, device,
                                batch_size)
            print(f"CLIP: {clip_score:.3f}")
        
        results.append([gen_img_dir, fid_score, clip_score])
        pd.DataFrame(results, columns=['gen_img_dir', 'fid', 'clip']).to_csv(f'/home/tien/project/Q-DiT/data/results_{gen_img_dir_name}.csv')


if __name__ == '__main__':
    main()

