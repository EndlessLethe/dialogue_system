'''
Author: Zeng Siwei
Date: 2020-08-27 17:05:13
LastEditors: Zeng Siwei
LastEditTime: 2020-08-29 20:45:08
Description: 
'''

import torch
import math, copy, time
import logging

def clones(module, N):
    "Produce N identical layers."
    return torch.nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class DotAttentionLayer(torch.nn.Module):
    def __init__(self, query_dim, key_dim, hidden_dim, dropout = 0.1):
        super(DotAttentionLayer, self).__init__()

        self.query_dim = query_dim
        self.key_dim = key_dim
        self.hidden_dim = hidden_dim
        self.W_q_layer = torch.nn.Linear(self.query_dim, self.hidden_dim)
        self.W_k_layer = torch.nn.Linear(self.key_dim, self.key_dim)
        self.dropout = torch.nn.Dropout(p=dropout)


    def forward(self, query, key, value, mask):
        '''
        Args: 
            query: shape (batch_size, 1, dim1)
            key == value: shape (batch_size, k, dim2)
            mask: shape (batch_size, 1, k)
        Returns: 
        '''
        logging.debug("DotAttentionLayer query shape: " + str(query.shape))
        logging.debug("DotAttentionLayer key shape: " + str(key.shape))

        query = self.W_q_layer(query)
        key = self.W_k_layer(key)
        scores = torch.matmul(query, key.transpose(-2, -1))

        logging.debug("DotAttentionLayer scores shape: " + str(scores.shape))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = torch.nn.functional.softmax(scores, dim = -1)

        logging.debug("DotAttentionLayer attention_vec shape: " + str(p_attn.shape))

        if self.dropout is not None:
            p_attn = self.dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn


def scaled_dot_self_attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = torch.nn.functional.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(torch.nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(torch.nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = torch.nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = scaled_dot_self_attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


if __name__ == "__main__":
    pass