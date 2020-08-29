import logging
import pandas as pd
import numpy as np
import jieba

class JDEmbeddingLoader():
    def __init__(self):
        self.data_skipgram_embedding = None
        self.dict_word2index = None

    def get_embedding(self, add_padding_token = True):
        if self.data_skipgram_embedding is None:
            self.load_embedding()

        self.data_skipgram_embedding.loc[self.data_skipgram_embedding.shape[0]] = [0] * self.data_skipgram_embedding.shape[1]
        self.dict_word2index["[PAD]"] = self.data_skipgram_embedding.shape[0]-1
        self.data_skipgram_embedding.loc[self.data_skipgram_embedding.shape[0]] = \
            np.random.uniform(0, 1, (self.data_skipgram_embedding.shape[1], ))
        self.dict_word2index["[EOS]"] = self.data_skipgram_embedding.shape[0] - 1
        self.data_skipgram_embedding.loc[self.data_skipgram_embedding.shape[0]] = \
            np.random.uniform(0, 1, (self.data_skipgram_embedding.shape[1],))
        self.dict_word2index["[BOS]"] = self.data_skipgram_embedding.shape[0] - 1
        self.data_skipgram_embedding.loc[self.data_skipgram_embedding.shape[0]] = \
            np.random.uniform(0, 1, (self.data_skipgram_embedding.shape[1],))
        self.dict_word2index["[OOV]"] = self.data_skipgram_embedding.shape[0] - 1
        logging.debug("skipgram embedding shape after adding padding token: " + str(self.data_skipgram_embedding.shape))
        logging.debug("padding token index is: " + str(self.dict_word2index["[PAD]"]))
        logging.debug("end token index is: " + str(self.dict_word2index["[EOS]"]))
        logging.debug("begin token index is: " + str(self.dict_word2index["[BOS]"]))
        logging.debug("oov token index is: " + str(self.dict_word2index["[OOV]"]))
        return self.data_skipgram_embedding, self.dict_word2index

    def fit(self, data):
        self.load_embedding()
        if isinstance(data, pd.DataFrame):
            corpus_skipgram = self.data2corpus(data)
        elif isinstance(data, str):
            corpus_skipgram = self.list_sentence_to_corpus([data])
        elif isinstance(data, list):
            corpus_skipgram = self.list_sentence_to_corpus(data)
        return corpus_skipgram

    def data2corpus(self, data):
        data.columns = [0]
        corpus_skipgram = self.list_sentence_to_corpus(data[0].values)
        return corpus_skipgram

    def list_sentence_to_corpus(self, list_sentence):
        corpus_list_sentence = []
        list_sentence_word_embedding = []
        cnt_empty = 0
        cnt_not_found = 0

        for i in range(len(list_sentence)):
            list_word_embedding = []
            sentence = list_sentence[i]
            sentence_cuted = list(jieba.cut(sentence))

            for j in range(len(sentence_cuted)):
                char_str = sentence_cuted[j]
                try:
                    index = self.dict_word2index[char_str]

                except KeyError:
                    cnt_not_found += 1
                    # logging.debug("char '" + char_str + "' is not in dict.")
                    continue

                word_embedding = list(self.data_skipgram_embedding.iloc[index])
                list_word_embedding.append(word_embedding)

            if len(list_word_embedding) != 0:
                corpus_sentence = []
                vec_sentence = np.sum(list_word_embedding, axis=0) / len(list_word_embedding)
                cnt = 0
                for vec in vec_sentence:
                    corpus_sentence.append((cnt, vec))
                    cnt += 1
            else:
                cnt_empty += 1
                corpus_sentence = []
            list_sentence_word_embedding.append(list_word_embedding)
            corpus_list_sentence.append(corpus_sentence)

            if i % 1000 == 0 and i != 0:
                logging.info("Finished {0} sentences.".format(i))

        logging.debug("corpus_skipgram size: " + str(len(corpus_list_sentence)))
        logging.debug("the first element in corpus_skipgram: " + str(corpus_list_sentence[0]))
        logging.debug("cnt_empty in corpus_skipgram: " + str(cnt_empty))
        logging.debug("cnt_not_found in corpus_skipgram: " + str(cnt_not_found))

        return corpus_list_sentence, list_sentence_word_embedding


    def load_embedding(self):
        logging.info("Loading skipgram embedding. Wait a moment.")
        filepath_skipgram_data = "./JDAI-WORD-EMBEDDING/JDAI-Word-Embedding.txt"
        data_skipgram = pd.read_csv(filepath_skipgram_data, sep=" ", skiprows=1, header=None)
        logging.debug(str(data_skipgram.iloc[0:10, 0:3]))

        data_skipgram.pop(301)
        data_word = data_skipgram.pop(0)
        logging.debug("skipgram embedding shape: " + str(data_skipgram.shape))
        dict_word2index = {}
        for i in range(data_word.shape[0]):
            dict_word2index[data_word[i]] = i
        logging.debug("dict_word2index is initialed properly: " + str(len(dict_word2index)))
        self.dict_word2index = dict_word2index
        self.data_skipgram_embedding = data_skipgram

if __name__ == '__main__':
    '''
    Usage:
        embedding_loader = JDEmbeddingLoader()
        data_embedding, dict_word2id = embedding_loader.get_embedding()
    '''
    
    logging.getLogger().setLevel(logging.DEBUG)

    embedding_loader = JDEmbeddingLoader()
    _, list_word_embedding = embedding_loader.fit("我的歌好听吗？以后会献上更好听的")
    print(len(list_word_embedding))
    print(len(list_word_embedding[0]))
    assert len(list_word_embedding) == 1
    assert len(list_word_embedding[0]) == 11
    data_embedding, dict_word2id = embedding_loader.get_embedding()