import torch

def extract_mask(embed, mask_idx, embedding_size, device):
    mask_embed = embed[mask_idx]
    mask_embed = torch.unsqueeze(mask_embed, 0)
    return mask_embed