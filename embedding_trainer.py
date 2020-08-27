import torch
import numpy as np
import logging
import pandas as pd
import jieba
from collections import defaultdict

# 使用词向量和attention后的字向量concat作为这个词的表示
# 对于两个句子求平均（？or跳过）
# 再将两个句子做一个self attention
# 将结果再过一个线性层做预测

n_word = None
max_len_utterance = 50
batch_size = 256
dim_embedding = 300
max_char_per_word = 5


def build_word_dict(filename):
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
            sentence = data.iat[i, j]
            sentence_cuted = list(jieba.cut(str(sentence)))
            for word in sentence_cuted:
                if word not in word_dict:
                    word_dict[word] = cnt
                    cnt += 1
                for ch in word:
                    if '\u4e00' <= ch <= '\u9fff':
                        word_dict[ch] = cnt
                        cnt += 1
                        char_dict[tag] = cnt_char
                        cnt_char += 1
    return word_dict, char_dict


def generate_negtive_sample(filepath_input, filepath_output, num_neg):
    """
    The format is "pair.a \t pair.b\n"
    """
    data_total = pd.read_csv(filepath_input, sep = "\t")
    logging.info("Total data size: " + str(data_total.shape[0]))

    n_pos = 0
    n_neg = 0
    with open(filepath_output, "w", encoding="utf-8") as f_out:
        for i in range(data_total.shape[0]):
                q = data_total.iat[i, 0]
                true_a = data_total.iat[i, 1]
                f_out.write(q + "\t" + true_a + "\t1\n")
                n_pos += 1

                cnt_neg = 0
                while cnt_neg < num_neg:
                    index_false_a = random.randint(0, data_total.shape[0])
                    false_a = data_total.iat[index_false_a, 1]
                    f_out.write(q + "\t" + false_a + "\t0\n")
                    n_neg += 1
            if (i+1) % 10000 == 0:
                logging.info("Finished {0} sentences.".format(i))

        logging.info("Generating Positive samples: " + str(n_pos))
        logging.info("Generating Negetive samples: " + str(n_neg))


    logging.info("output candidate file to:" + filepath_output)


def get_dataloader(filename, dict_word2id, dict_char2id):
    """
    File should have the format as: sentence1 \t sentence2\n
    Using neg sample to generate neg samples
    """

    data = pd.read_csv(filename, header=None, sep="\t")
    logging.info(filename + "\n" + str(data.iloc[0:10]))
    y = np.array(data.pop(data.shape[1] - 1))
    logging.info("label: " + str(y[0:10]))

    list_input = []
    x_data = np.zeros((data.shape[0], data.shape[1], max_len_utterance))
    x_data_char = np.zeros((data.shape[0], data.shape[1], max_len_utterance, max_char_per_word))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            sentence = data.iat[i, j]
            sentence_cuted = list(jieba.cut(str(sentence)))
            x_data[i][j][0] = dict_word2id["[BOS]"]
            cnt = 1
            for word in sentence_cuted:
                if cnt >= max_len_utterance - 1:
                    break
                if word in dict_word2id:
                    x_data[i][j][cnt] = dict_word2id[word]
                else:
                    x_data[i][j][cnt] = dict_word2id["[OOV]"]
                if is_char:
                    cnt_char = 0
                    for ch in word:
                        if cnt_char >= max_char_per_word:
                            break
                        if '\u4e00' <= ch <= '\u9fff':
                            x_data_char[i][j][cnt][cnt_char] = dict_char2id[ch]
                            cnt_char += 1
                    while (cnt_char < max_char_per_word):
                        x_data_char[i][j][cnt][cnt_char] = dict_char2id["[PAD]"]
                        cnt_char += 1
                cnt += 1
            while (cnt < max_len_utterance - 1):
                x_data[i][j][cnt] = dict_word2id["[PAD]"]
                cnt += 1
            x_data[i][j][cnt] = dict_word2id["[EOS]"]
            list_input.append([x_data, x_data_char])

    dataset = torch.utils.data.TensorDataset(torch.from_numpy(np.array(list_input, copy=false)).long(), torch.from_numpy(y).long())
    dataset = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataset


class CharAndWordEmbedding(torch.nn.Module):
    """
    This is used as an embedding layer.
    """
    def __init__(self, n_dict_char, n_dict_word, char_pretrained_embedding = None, word_pretrained_embedding = None):
        self.char_embedding = torch.nn.Embedding(num_embeddings=n_dict_char, embedding_dim=dim_embedding, padding_idx=0)
        self.word_embedding = torch.nn.Embedding(num_embeddings=word_pretrained_embedding, embedding_dim=dim_embedding, padding_idx=0)

        ### init layers
        if char_pretrained_embedding is not None:
            self.char_embedding.weight = torch.nn.Parameter(torch.tensor(char_pretrained_embedding, dtype=torch.float32))
            self.char_embedding.weight.requires_grad = True
        else :
            # torch.nn.init.uniform_(self.embedding_layer.weight)
            pass # use default setting

        if char_pretrained_embedding is not None:
            self.word_embedding.weight = torch.nn.Parameter(torch.tensor(word_pretrained_embedding, dtype=torch.float32))
            self.word_embedding.weight.requires_grad = True
        else :
            # torch.nn.init.uniform_(self.embedding_layer.weight)
            pass # use default setting

    def forward(word_seq, list_char_seq):
        vec_list_char = self.embedding_layer(char_seq)

        # (batch_size, batch_sentence_len, dim_embedding)
        vec_word = self.embedding_layer(word_seq)

        vec_char = torch.sum(vec_list_char, dim=0) 

        vec_cat = torch.cat(vec_word, vec_char)