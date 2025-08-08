import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
import numpy as np
from glob import glob

IMAGENET_DIR = './imagenet/content/'
STYLE_IMAGENET_DIR = './imagenet/style/'
SEG_DIR = './imagenet/content-mask/'
STYLE_SEG_DIR = './imagenet/style-mask/'

COLOR_CODES = {
    'UnAttack': lambda r, g, b: (r < 0.5) & (g < 0.5) & (b < 0.5),
    'Attack': lambda r, g, b: (r > 0.8) & (g > 0.8) & (b > 0.8),
}

img_transform = T.Compose([
    T.Resize((400, 400)),
    T.ToTensor()
])

class AdvDataset(Dataset):
    def __init__(self, results):
        self.data = results
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def extract_mask(seg_tensor: torch.Tensor, color_str: str):
    r, g, b = seg_tensor[0], seg_tensor[1], seg_tensor[2]
    mask = COLOR_CODES[color_str](r, g, b).float()
    return mask.unsqueeze(0)

def load_images_with_mask():
    
    # Find all jpg images in the directory and its subdirectories
    image_paths = glob(os.path.join(IMAGENET_DIR, '**', '*.jpg'), recursive=True)
    results = []
    
    for path in image_paths:
        filename = os.path.basename(path)
        style_path = f'{STYLE_IMAGENET_DIR}/{filename}'
        seg_path = f'{SEG_DIR}/{filename}'
        style_seg_path = f'{STYLE_SEG_DIR}/{filename}'

        img = Image.open(path).convert('RGB')
        img_tensor = img_transform(img)              # [3, 400, 400]
        style_img = Image.open(style_path).convert('RGB') 
        style_img_tensor = img_transform(style_img)  # [3, 400, 400]

        seg_img = Image.open(seg_path).convert('RGB').resize((img_tensor.shape[1], img_tensor.shape[2]), resample=Image.BILINEAR)
        seg_array = np.array(seg_img).astype(np.float32) / 245.0
        seg_tensor = torch.from_numpy(seg_array).permute(2, 0, 1).contiguous()

        style_seg_img = Image.open(style_seg_path).convert('RGB').resize((style_img_tensor.shape[1], style_img_tensor.shape[2]), resample=Image.BILINEAR)
        style_seg_array = np.array(style_seg_img).astype(np.float32) / 245.0
        style_seg_tensor = torch.from_numpy(style_seg_array).permute(2, 0, 1).contiguous() # [3, 400, 400]

        # Generate attack and unattack masks
        content_mask_attack = extract_mask(seg_tensor, 'Attack')
        content_mask_unattack = extract_mask(seg_tensor, 'UnAttack')

        style_mask_attack = extract_mask(style_seg_tensor, 'Attack')
        style_mask_unattack = extract_mask(style_seg_tensor, 'UnAttack')

        results.append({
            'image':img_tensor,
            'style_image':style_img_tensor,
            'mask_unattack': content_mask_unattack,
            'mask_attack': content_mask_attack,
            'style_mask_unattack': style_mask_unattack,
            'style_mask_attack': style_mask_attack,
        })

    adv_dataset = AdvDataset(results)

    return adv_dataset
        


    
    
