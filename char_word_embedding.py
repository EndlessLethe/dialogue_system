'''
Author: Zeng Siwei
Date: 2020-08-27 17:05:13
LastEditors: Zeng Siwei
LastEditTime: 2020-08-30 00:29:38
Description: 
'''

import torch
import numpy as np
import logging
import pandas as pd
import jieba
# from collections import defaultdict
from function_utils import *
from layer_utils import *

# 使用词向量和attention后的字向量concat作为这个词的表示

def process_sentence(sentence, dict_word2id, dict_char2id = dict(), max_len_utterance = 50, max_char_per_word = 5, use_char = False):
    x_data = np.zeros(max_len_utterance)
    if use_char:
        x_data_char = np.zeros((max_len_utterance, max_char_per_word))
    sentence_cuted = list(jieba.cut(str(sentence)))
    x_data[0] = dict_word2id["[BOS]"]
    cnt = 1
    for word in sentence_cuted:
        if cnt >= max_len_utterance - 1:
            break
        if word in dict_word2id:
            x_data[cnt] = dict_word2id[word]
        else:
            x_data[cnt] = dict_word2id["[OOV]"]
        if use_char:
            cnt_char = 0
            for ch in word:
                if cnt_char >= max_char_per_word:
                    break
                if '\u4e00' <= ch <= '\u9fff':
                    x_data_char[cnt][cnt_char] = dict_char2id[ch]
                    cnt_char += 1
            while (cnt_char < max_char_per_word):
                x_data_char[cnt][cnt_char] = dict_char2id["[PAD]"]
                cnt_char += 1
        cnt += 1
    while (cnt < max_len_utterance - 1):
        x_data[cnt] = dict_word2id["[PAD]"]
        cnt += 1
    x_data[cnt] = dict_word2id["[EOS]"]
    if use_char:
        return x_data, x_data_char
    else :
        return x_data

def get_char_word_dataloader(filename, dict_word2id, dict_char2id, max_len_utterance, max_char_per_word, batch_size = 256, cols = []):
    """
    File should have the format as: sentence1 \t sentence2 ... \t ... sentencek \n
    """

    data = pd.read_csv(filename, header=None, sep="\t")
    logging.info(filename + "\n" + str(data.iloc[0:10]))
    y = np.array(data.pop(data.shape[1] - 1))
    logging.info("label: " + str(y[0:10]))

    n_sentence = len(cols) if cols else data.shape[1]
    x_data = np.zeros((data.shape[0], n_sentence, max_len_utterance))
    x_data_char = np.zeros((data.shape[0], n_sentence, max_len_utterance, max_char_per_word))
    for i in range(data.shape[0]):
        cnt = 0
        for j in range(data.shape[1]):
            if cols and j not in cols:
                continue
            sentence = data.iat[i, j]
            x_data[i][cnt], x_data_char[i][cnt] = process_sentence(sentence, dict_word2id, dict_char2id, use_char = True)
            cnt += 1

    class CharAndWordSeqDataset(torch.utils.data.Dataset):
        def __init__(self, x_data, x_data_char, y):
            self.x_data = x_data
            self.x_data_char = x_data_char
            self.y = y

        def __len__(self):
            return self.x_data.shape[0]

        def __getitem__(self, idx):
            return [x_data[idx], x_data_char[idx]], y[idx]

    dataset = CharAndWordSeqDataset(x_data, x_data_char, y)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


class CharAndWordEmbedding(torch.nn.Module):
    """
    This is used as an embedding layer.
    """
    def __init__(self, n_dict_char, n_dict_word, embedding_dim, char_pretrained_embedding = None, word_pretrained_embedding = None):
        super(CharAndWordEmbedding, self).__init__()
        
        self.char_embedding = torch.nn.Embedding(num_embeddings=n_dict_char, embedding_dim=embedding_dim, padding_idx=0)
        self.word_embedding = torch.nn.Embedding(num_embeddings=n_dict_word, embedding_dim=embedding_dim, padding_idx=0)
        self.attention_layer = DotAttentionLayer(embedding_dim, embedding_dim, embedding_dim)

        ### init layers
        if char_pretrained_embedding is not None:
            self.char_embedding.weight = torch.nn.Parameter(torch.tensor(char_pretrained_embedding, dtype=torch.float32))
            self.char_embedding.weight.requires_grad = True
        else :
            # torch.nn.init.uniform_(self.char_embedding.weight)
            pass # use default setting

        if char_pretrained_embedding is not None:
            self.word_embedding.weight = torch.nn.Parameter(torch.tensor(word_pretrained_embedding, dtype=torch.float32))
            self.word_embedding.weight.requires_grad = True
        else :
            # torch.nn.init.uniform_(self.word_embedding.weight)
            pass # use default setting

    def forward(self, word_seq, list_char_seq):
        '''
        Args: 
            word_seq: shape (batch_size, max_sentence_len)  
            list_char_seq: shape (batch_size, max_sentence_len, max_char_len)
        Returns: 
		
        '''
        logging.debug("CharAndWordEmbedding word_seq shape: " + str(word_seq.shape))
        logging.debug("list_char_seq shape: " + str(list_char_seq.shape))

        ## (batch_size, max_sentence_len, 1, max_char_len)
        char_mask = padding_mask(list_char_seq, 0)
        logging.debug("CharAndWordEmbedding char_mask shape: " + str(char_mask.shape))

        ## (batch_size, max_sentence_len, 1, embedding_dim)
        vec_word = self.word_embedding(word_seq)
        vec_word = vec_word.unsqueeze(-2)
        logging.debug("CharAndWordEmbedding vec_word shape: " + str(vec_word.shape))
 
        ## (batch_size, max_sentence_len, max_char_len, embedding_dim)
        vec_list_char = self.char_embedding(list_char_seq)
        logging.debug("CharAndWordEmbedding vec_list_char shape: " + str(vec_list_char.shape))
        

        # vec_char = torch.sum(vec_list_char, dim=0) 
        vec_char, _ = self.attention_layer(vec_word, vec_list_char, vec_list_char, char_mask)
        vec_cat = torch.cat((vec_word, vec_char), dim=-1)
        vec_cat = vec_cat.squeeze(-2)
        logging.debug("CharAndWordEmbedding vec_cat shape: " + str(vec_cat.shape))
        return vec_cat
        

def test_data():
    word_dict, char_dict = build_word_dict("./data/simtrain_to05sts.txt", [1, 3])
    x_data, x_data_char = process_sentence("你的名字是什么鸭", word_dict, char_dict, use_char = True)
    print(x_data)
    print(x_data_char)
    tensor = torch.Tensor(x_data_char).long()
    char_mask = padding_mask(tensor, 0)
    print(char_mask.shape)

def test_embedding(use_gpu = False):
    word_dict, char_dict = build_word_dict("./data/simtrain_to05sts.txt", [1, 3])
    dataloader = get_char_word_dataloader("./data/simtrain_to05sts.txt", word_dict, char_dict, 50, 5, cols = [1, 3])

    for i, data in enumerate(dataloader):
        inputs, labels = data
        for i in range(len(inputs)):
            inputs[i] = inputs[i].long()
            print(inputs[i])
            print(inputs[i].dtype)
            print(inputs[i].shape)

        char_word_embedding_layer = CharAndWordEmbedding(len(word_dict), len(char_dict), 300)

        if use_gpu:
            inputs = [x.cuda() for x in inputs]
            labels = labels.cuda()
            char_word_embedding_layer = char_word_embedding_layer.cuda()
        
        
        word_seq = inputs[0].transpose(0, 1)
        sentence1_word = word_seq[0]
        sentence2_word = word_seq[1]

        char_seq = inputs[1].transpose(0, 1)
        sentence1_char = char_seq[0]
        sentence2_char = char_seq[1]

        vec_char_word = char_word_embedding_layer(sentence1_word, sentence1_char)
        print(vec_char_word)
        break


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    logging.getLogger("jieba").setLevel(logging.INFO)
    
    # test_data()
    test_embedding()
    