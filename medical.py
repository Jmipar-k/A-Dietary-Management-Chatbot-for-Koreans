import os
import random
import numpy as np 
import torch
import torch.nn as nn
import clip
import argparse  
from typing import Dict
from medical_layers import LoRALayer, PlainMultiheadAttentionLoRA

INDEX_POSITIONS_TEXT = {
    'top1': [11],
    'top2': [10, 11],
    'top3': [9, 10, 11],
    'bottom': [0, 1, 2, 3],
    'mid': [4, 5, 6, 7],
    'up': [8, 9, 10, 11],
    'half-up': [6, 7, 8, 9, 10, 11],
    'half-bottom': [0, 1, 2, 3, 4, 5],
    'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]}

INDEX_POSITIONS_VISION = {
    'ViT-B/16': {
        'top': [11],
        'top3': [9, 10, 11],
        'bottom': [0, 1, 2, 3],
        'mid': [4, 5, 6, 7],
        'up': [8, 9, 10, 11],
        'half-up': [6, 7, 8, 9, 10, 11],
        'half-bottom': [0, 1, 2, 3, 4, 5],
        'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]},
    'ViT-B/32': {
        'bottom': [0, 1, 2, 3],
        'mid': [4, 5, 6, 7],
        'up': [8, 9, 10, 11],
        'half-up': [6, 7, 8, 9, 10, 11],
        'half-bottom': [0, 1, 2, 3, 4, 5],
        'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]},

    'ViT-L/14': {
        'half-up': [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
        'half-bottom': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]}
}

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def apply_lora(clip_model):
    list_lora_layers = []
    # text encoder
    indices = INDEX_POSITIONS_TEXT['all']
    text_encoder = clip_model.transformer
    for i, block in enumerate(text_encoder.resblocks):
        print(f"Residual Attention Block {i}: {block}")
        if i in indices:
            for name, submodule in block.named_children():
                if isinstance(submodule, nn.MultiheadAttention):
                    new_multi_head_lora = PlainMultiheadAttentionLoRA(submodule)
                    setattr(block, name, new_multi_head_lora)
                    list_lora_layers.append(new_multi_head_lora)

    # image encoder
    indices = INDEX_POSITIONS_VISION['ViT-B/16']['all']
    vision_encoder = clip_model.visual.transformer
    for i, block in enumerate(vision_encoder.resblocks):
        print(f"Residual Attention Block {i}: {block}")
        if i in indices:
            for name, submodule in block.named_children():
                if isinstance(submodule, nn.MultiheadAttention):
                    new_multi_head_lora = PlainMultiheadAttentionLoRA(submodule)
                    setattr(block, name, new_multi_head_lora)
                    list_lora_layers.append(new_multi_head_lora)
    return list_lora_layers

def load_lora(list_lora_layers, ckpt_path):
    # to manage names like ViT-B/16
    backbone = 'ViT-B/16'.replace('/', '').replace('-', '').lower()
    load_path = ckpt_path

    if not os.path.exists(load_path):
        raise FileNotFoundError(f'File {load_path} does not exist.')

    loaded_data = torch.load(load_path)

    metadata = loaded_data['metadata']
    if metadata['r'] != 2:
        raise ValueError(
            f"r mismatch: expected {2}, found {metadata['r']}")
    if metadata['alpha'] != 1:
        raise ValueError(
            f"alpha mismatch: expected {1}, found {metadata['alpha']}")
    if metadata['encoder'] != 'both':
        raise ValueError(
            f"Encoder mismatch: expected {'both'}, found {metadata['encoder']}")
    if metadata['params'] != ['q', 'k', 'v']:
        raise ValueError(
            f"Params mismatch: expected {['q', 'k', 'v']}, found {metadata['params']}")
    if metadata['position'] != 'all':
        raise ValueError(
            f"Position mismatch: expected {'all'}, found {metadata['position']}")

    weights = loaded_data['weights']
    for i, layer in enumerate(list_lora_layers):
        layer_weights = weights[f'layer_{i}']
        if 'q_proj' in layer_weights:
            layer.q_proj.w_lora_A.data.copy_(
                layer_weights['q_proj']['w_lora_A'])
            layer.q_proj.w_lora_B.data.copy_(
                layer_weights['q_proj']['w_lora_B'])
        if 'k_proj' in layer_weights:
            layer.k_proj.w_lora_A.data.copy_(
                layer_weights['k_proj']['w_lora_A'])
            layer.k_proj.w_lora_B.data.copy_(
                layer_weights['k_proj']['w_lora_B'])
        if 'v_proj' in layer_weights:
            layer.v_proj.w_lora_A.data.copy_(
                layer_weights['v_proj']['w_lora_A'])
            layer.v_proj.w_lora_B.data.copy_(
                layer_weights['v_proj']['w_lora_B'])

    print(f'LoRA weights loaded from {load_path}')

def main(ckpt_path):
    
    clip_model, preprocess = clip.load('ViT-B/16')
    clip_model.eval()

    list_lora_layers = apply_lora(clip_model)
    clip_model = clip_model.cuda() 
    
    load_lora(list_lora_layers, ckpt_path)
    return clip_model

def lora_clip():
    set_random_seed(1)
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, default="./lora_weights.pt")
    args = parser.parse_args()
    clip_model, preprocess = clip.load('ViT-B/16')
    clip_model.eval()

    list_lora_layers = apply_lora(clip_model)
    clip_model = clip_model.cuda()
    load_lora(list_lora_layers, "./lora_weights.pt")
    return clip_model, preprocess
# if __name__ == '__main__':
#     set_random_seed(1)
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--ckpt_path', type=str, default="./vitb16/food/16shots/seed1/lora_weights.pt")
#     args = parser.parse_args()
#     main(args.ckpt_path)
