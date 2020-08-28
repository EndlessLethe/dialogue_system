'''
Author: Zeng Siwei
Date: 2020-08-28 16:09:21
LastEditors: Zeng Siwei
LastEditTime: 2020-08-28 19:41:28
Description: 
'''
import torch

def padding_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)   # [B, 1, L]

# def subsequent_mask(size):
#     "Mask out subsequent positions."
#     attn_shape = (1, size, size)
#     subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
#     return torch.from_numpy(subsequent_mask) == 0

def generate_square_subsequent_mask(self, sz):
    r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask