import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import models
from torchvision import transforms
from torchvision.transforms.functional import resize
import os
import math

from loss import ContentLoss, StyleLoss, adv_loss_fn, gram_matrix, tv_loss
from dataset import load_images_with_mask


class VGGFeatureExtractor(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.vgg = models.vgg19(pretrained=True).features.eval()
        self.layers = layers
        self.outputs = {}

        def hook_fn(name):
            def hook(module, input, output):
                self.outputs[name] = output
            return hook

        self.name2idx = {
            'conv1_1': 0,
            'conv2_1': 5,
            'conv3_1': 10,
            'conv4_1': 19,
            'conv4_2': 21,
            'conv5_1': 28,
        }

        for name in layers:
            self.vgg[self.name2idx[name]].register_forward_hook(hook_fn(name))

    def forward(self, x):
        self.vgg(x)
        return {name: self.outputs[name] for name in self.layers}
        

def compute_masked_gram(feature, mask):
    # feature: [B, C, H, W], mask: [B, 1, H, W]
    masked_feat = feature * mask
    B, C, H, W = masked_feat.shape
    mask_mean = mask.mean()
    if mask_mean.item() == 0:
        return None, 0
    # gram = gram_matrix(masked_feat)
    # gram = gram / (C * H * W)
    F_flat = masked_feat.view(B, C, -1)
    M_flat = mask.view(B, 1, -1)
    # 计算有效像素数（每个样本独立）
    N_eff = M_flat.sum(dim=2, keepdim=True).clamp_min(1.0)  # 避免除0
    # 加权特征：F * sqrt(M) 让 Gram 近似 E[F F^T] over masked region
    Fw = F_flat * (M_flat.sqrt())
    G = torch.bmm(Fw, Fw.transpose(1, 2)) / N_eff 
    
    return G, mask_mean

def style_loss_layerwise(feat_adv, feat_const, feat_content_const, content_masks, style_masks, CNN_structure, layer_names):
    total_loss = 0.0
    c_masks = content_masks.copy()
    s_masks = style_masks.copy()
    content_seg_height = content_masks[0].shape[2]
    content_seg_width = content_masks[0].shape[3]
    style_seg_height = style_masks[0].shape[2]
    style_seg_width = style_masks[0].shape[3]
    for name in CNN_structure:
        
        if "pool" in name:
            content_seg_height = int(math.ceil(content_seg_height / 2))
            content_seg_width = int(math.ceil(content_seg_width / 2))
            style_seg_height = int(math.ceil(style_seg_height / 2))
            style_seg_width = int(math.ceil(style_seg_width / 2))
            for i in range(len(c_masks)):
                c_masks[i] = F.interpolate(c_masks[i], size=(content_seg_height, content_seg_width), mode='bilinear', align_corners=False)
                s_masks[i] = F.interpolate(s_masks[i], size=(style_seg_height, style_seg_width), mode='bilinear', align_corners=False)
        # elif "conv" in name:
        #     for i in range(len(c_masks)):
        #         c_masks[i] = F.avg_pool2d(c_masks[i], kernel_size=3, stride=1, padding=1)
        #         s_masks[i] = F.avg_pool2d(s_masks[i], kernel_size=3, stride=1, padding=1)
        
                
        if name in layer_names:
            f_adv = feat_adv[name]       # variable feature
            f_const = feat_const[name]   # target style
            f_cont_const = feat_content_const[name]  # fallback style when s_mask == 0

            for c_mask, s_mask in zip(c_masks, s_masks):
                G_adv, c_mean = compute_masked_gram(f_adv, c_mask)
                if G_adv is None:
                    continue

                s_mean = s_mask.mean()
                use_f_const = f_cont_const if (s_mean.item() == 0 and c_mean.item() > 0) else f_const
                use_mask = c_mask if (s_mean.item() == 0 and c_mean.item() > 0) else s_mask

                G_target, _ = compute_masked_gram(use_f_const, use_mask)
                if G_target is None:
                    continue

                diff = F.mse_loss(G_adv, G_target, reduction='mean') * c_mean
                total_loss += diff
                
    return total_loss


def style_attack(args, model, feature_extractor, batch, batch_id, save=True):
    x_orig = batch['image'].to(args.device)
    x_style = batch['style_image'].to(args.device)
    x_adv = x_orig.clone().detach().requires_grad_(True)
    IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], device=args.device).view(1,3,1,1)
    IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225], device=args.device).view(1,3,1,1)

    def normalize(x):
        return (x - IMAGENET_MEAN) / IMAGENET_STD


    mask_attack = batch['mask_attack'].to(args.device)
    mask_unattack = batch['mask_unattack'].to(args.device)
    style_mask_attack = batch['style_mask_attack'].to(args.device)
    layer_names = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv4_2', 'conv5_1']
    CNN_structure = [
    'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
    'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
    'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
    'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
    'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5'
]

    with torch.no_grad():
        x_in_orig = normalize(x_orig)
        x_in_style = normalize(x_style)
        feat_orig = feature_extractor(x_in_orig)
        feat_style = feature_extractor(x_in_style)

    optimizer = Adam([x_adv], lr=args.lr)

    for step in range(args.num_steps):
        optimizer.zero_grad()

        # content loss
        x_in = normalize(x_adv)
        feat_adv = feature_extractor(x_in)
        f_c_adv = feat_adv['conv4_2'] * resize(mask_unattack, feat_adv['conv4_2'].shape[2:])
        f_c_orig = feat_orig['conv4_2'] * resize(mask_unattack, feat_adv['conv4_2'].shape[2:])
        # f_c_adv = feat_adv['conv4_2']
        # f_c_orig = feat_orig['conv4_2']
        content_loss_fn = ContentLoss(f_c_orig)
        content_loss = content_loss_fn(f_c_adv)
         
        # style loss
        style_loss = style_loss_layerwise(feat_adv, feat_style, feat_orig, content_masks=[mask_attack], style_masks=[style_mask_attack], CNN_structure=CNN_structure, layer_names=layer_names)

        # adv loss
        logits = model(x_in)
        adv_loss = adv_loss_fn(logits, torch.tensor([args.target_label], device=args.device))

        # total loss
        total_loss = args.content_weight * content_loss + args.style_weight * style_loss + args.adv_weight * adv_loss
        total_loss.backward()
        optimizer.step()

        x_adv.data = x_adv.data.clamp(0, 1)
        if step % 50 == 0:
            # print(f'Step {step}/{args.num_steps}, Total Loss: {total_loss.item():.4f}, Content Loss: {content_loss.item():.4f}, Style Loss: {style_loss.item():.4f}, Adv Loss: {adv_loss.item():.4f}')
            print(f'Step {step}/{args.num_steps}, Total Loss: {total_loss.item()}, Content Loss: {content_loss.item()}, Style Loss: {style_loss.item()}, Adv Loss: {adv_loss.item()}, TV Loss: {tv_loss(x_adv).item()}')
            if save:
                output_path = f'{args.result_dir}/{batch_id}/{step}.png'
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                adv_img = transforms.ToPILImage()(x_adv.squeeze(0).cpu())
                adv_img.save(output_path)
    
    # Evaluate the adversarial image with the model
    with torch.no_grad():
        x_in = normalize(x_adv)
        logits = model(x_in)
        probs = F.softmax(logits, dim=1)
        pred_label = torch.argmax(logits, dim=1).item()
        target_prob = probs[0, args.target_label].item()
        
        print(f"Attack finished - Prediction: {pred_label}, Target: {args.target_label}")
        print(f"Confidence for target class: {target_prob:.4f}")
        
        if pred_label == args.target_label:
            print("Attack successful! Model classified the image as the target class.")
        else:
            print("Attack unsuccessful. Model did not classify the image as the target class.")
        
    return x_adv.detach()

def run_attack(args):
    result_dir = args.result_dir if hasattr(args, 'result_dir') else 'results'
    os.makedirs(result_dir, exist_ok=True)
    dataset = load_images_with_mask()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    feature_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv4_2', 'conv5_1']
    vgg_extractor = VGGFeatureExtractor(feature_layers).to(args.device)
    model = models.vgg19(pretrained=True).to(args.device)
    model.eval()
    for i, batch in enumerate(dataloader):
        x_adv = style_attack(args, model, vgg_extractor, batch, i)
        # Save the adversarial image
        # Convert tensor to PIL image and save
        adv_img = transforms.ToPILImage()(x_adv.squeeze(0).cpu())
        filename = f'adv_image_{i}.png'
        output_path = os.path.join(result_dir, filename)
        adv_img.save(output_path)
        print(f"Saved adversarial image to {output_path}")


