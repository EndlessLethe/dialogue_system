'''
Author: Zeng Siwei
Date: 2020-08-28 16:09:21
LastEditors: Zeng Siwei
LastEditTime: 2020-08-29 20:46:58
Description: 
'''
import torch
import pandas as pd
import logging
import jieba

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

def build_word_dict(filename, cols=[]):
    '''
    Args: 
	
    Returns: 
	
    Usage:
        word_dict, char_dict = build_word_dict("./data/simtrain_to05sts.txt", [1, 3])
    '''
    word_dict = dict()
    char_dict = dict()

    # 后续添加列参数，只取文件某一列的数据
    data = pd.read_csv(filename, header=None, sep="\t")
    logging.info(filename + "\n" + str(data.iloc[0:10]))

    PRE_TAGS = ["[PAD]", "[OOV]", "[BOS]", "[EOS]"]
    cnt = 0
    cnt_char = 0
    for tag in PRE_TAGS:
        word_dict[tag] = cnt
        cnt += 1
        char_dict[tag] = cnt_char
        cnt_char += 1

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if not cols or j in cols:
                sentence = data.iat[i, j]
                sentence_cuted = list(jieba.cut(str(sentence)))
                for word in sentence_cuted:
                    if word not in word_dict:
                        word_dict[word] = cnt
                        cnt += 1
                    for ch in word:
                        if u'\u4e00' <= ch <= u'\u9fff':
                            if ch not in word_dict:
                                word_dict[ch] = cnt
                                cnt += 1
                            if ch not in char_dict:
                                char_dict[ch] = cnt_char
                                cnt_char += 1
    return word_dict, char_dict