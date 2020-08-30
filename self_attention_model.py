'''
Author: Zeng Siwei
Date: 2020-08-29 20:15:21
LastEditors: Zeng Siwei
LastEditTime: 2020-08-30 08:00:08
Description: 
'''

import torch
from char_word_embedding import *

# 将两个句子做一个self attention(过一层transformer)
# 将结果再过一个线性层做预测

class SelfAttentionModel(torch.nn.Module):
    def __init__(self, n_dict_char, n_dict_word, embedding_dim):
        super(SelfAttentionModel, self).__init__()

        ## embedding_dim = dim_char + dim_word
        self.embedding_layer = CharAndWordEmbedding(n_dict_char, n_dict_word, embedding_dim // 2)
        self.attention_layer = MultiHeadedAttention(1, embedding_dim)
        self.linear_layer = torch.nn.Linear(embedding_dim, 2)

    def forward(self, inputs):

        
        word_seq = inputs[0].transpose(0, 1)
        sentence1_word = word_seq[0]
        sentence2_word = word_seq[1]

        char_seq = inputs[1].transpose(0, 1)
        sentence1_char = char_seq[0]
        sentence2_char = char_seq[1]

        # (batch_size, max_sentence_len, dim)
        vec_sentence_a = char_word_embedding_layer(sentence1_word, sentence1_char)
        vec_sentence_b = char_word_embedding_layer(sentence2_word, sentence2_char)

        vec_b_attention = self.attention_layer(vec_sentence_a, vec_sentence_b, vec_sentence_b)

        logits = self.linear_layer(last_hidden)
        y_pred = logits
        return logits


def test_model(use_gpu = False):
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
    test_model()