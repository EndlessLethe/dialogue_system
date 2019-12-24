import torch
import numpy as np
from jd_embedding_loader import JDEmbeddingLoader
import logging
import pandas as pd
import jieba

batch_size = 512
n_dict = 1054344
dim_embedding = 300 # 200 dim trained with word2vec in passage, here use char-
dim_gru1 = 300 # 200 in passage
dim_gru2 = 100 # 50 in passage
n_in_channel = 2 # word and segment matrix
n_feature_map = 8 # 8 in passage
kernel_shape = (3, 3) # (3, 3) in passage
magic_3 = 3 # use to control cnn arg and output shape
pool_shape = (magic_3, magic_3) # (3, 3) in passage
pool_stride = (magic_3, magic_3) # (3, 3) in passage
gru_dorpout = 0
dim_sim_vec = 50


learning_rate = 1e-4
max_len_utterance = 38 # make sure (max_len_utterance - 2) % 3 == 0
batch_sentence_len = max_len_utterance # here batch len must be a constant to make sure the shape of feature map
n_utterance = 3
n_label = 2


class SMNModel(torch.nn.Module):
    """
    This model is supposed to use pre-trained embedding

    For each sample:
    Input - list_history and query represented by ids
    Output - 2 class softmax result

    """
    def __init__(self, pretrained_embedding = None, use_gpu = False):
        super(SMNModel, self).__init__()

        ### define layers

        ## load embedding from file
        self.embedding_layer = torch.nn.Embedding(num_embeddings=n_dict, embedding_dim=dim_embedding)
        self.gru1_layer = torch.nn.GRU(input_size=dim_embedding, hidden_size=dim_gru1, batch_first=True,
                                      dropout=gru_dorpout, bidirectional=False)

        # instead of bilineaer_layer, we use a linear transform matrix instead
        # self.bilinear_layer = torch.nn.Bilinear(in1_features=dim_gru1, in2_features=)

        self.conv1_layer = torch.nn.Conv2d(in_channels=n_in_channel, out_channels=n_feature_map, kernel_size=kernel_shape)
        self.pool_layer = torch.nn.MaxPool2d(kernel_size=pool_shape, stride=pool_stride)

        ## only one-layer cnn
        # self.conv2_layer = torch.nn.Conv2d(in_channels=n_feature_map, out_channels=n_feature_map, kernel_size=kernel_shape)

        ## flatten all matrices of h'u and h'r as vi
        assert ((batch_sentence_len-magic_3+1) % magic_3 == 0 )
        self.cnn_linear_layer = torch.nn.Linear(in_features=int((batch_sentence_len-magic_3+1)**2 * n_feature_map / magic_3**2 ), out_features=dim_sim_vec)
        self.gru2_layer = torch.nn.GRU(input_size=dim_sim_vec, hidden_size=dim_gru2, batch_first=True,
                                       dropout=gru_dorpout, bidirectional=False)

        ## input the concat of all h'u and h'r, output matching socre
        ## In this paper, there's three strategies.
        self.gru_linear_layer = torch.nn.Linear(in_features=dim_gru2, out_features=n_label)

        ### init layers
        if pretrained_embedding is not None:
            self.embedding_layer.weight = torch.nn.Parameter(torch.tensor(pretrained_embedding, dtype=torch.float32))
            self.embedding_layer.weight.requires_grad = False
        else :
            # torch.nn.init.uniform_(self.embedding_layer.weight)
            pass # use default setting

        ih_gru1 = (param.data for name, param in self.gru1_layer.named_parameters() if 'weight_ih' in name)
        hh_gru1 = (param.data for name, param in self.gru1_layer.named_parameters() if 'weight_hh' in name)
        for w in ih_gru1:
            torch.nn.init.orthogonal_(w)
        for w in hh_gru1:
            torch.nn.init.orthogonal_(w)

        self.linear_transfom_op = torch.ones((dim_gru1, dim_gru1), requires_grad=True)
        if use_gpu:
            self.linear_transfom_op = self.linear_transfom_op.cuda()

        conv1_weight = (param.data for name, param in self.conv1_layer.named_parameters() if "weight" in name)
        for w in conv1_weight:
            torch.nn.init.kaiming_normal_(w)

        cnn_linear_weight = (param.data for name, param in self.cnn_linear_layer.named_parameters() if "weight" in name)
        for w in cnn_linear_weight:
            torch.nn.init.xavier_uniform_(w)

        ih_gru2 = (param.data for name, param in self.gru2_layer.named_parameters() if 'weight_ih' in name)
        hh_gru2 = (param.data for name, param in self.gru2_layer.named_parameters() if 'weight_hh' in name)
        for w in ih_gru2:
            torch.nn.init.orthogonal_(w)
        for w in hh_gru2:
            torch.nn.init.orthogonal_(w)

        gru_linear_weight = (param.data for name, param in self.gru_linear_layer.named_parameters() if "weight" in name)
        for w in gru_linear_weight:
            torch.nn.init.xavier_uniform_(w)

    def forward(self, list_history, response):
        """
            utterance:(n_utterance, batch_size, batch_sentence_len)
            response:(batch_size, batch_sentence_len)
        """

        ### build model

        # (n_utterance-1, batch_size, batch_sentence_len, dim_embedding)
        vec_word_list_history = self.embedding_layer(list_history)

        # (batch_size, batch_sentence_len, dim_embedding)
        vec_word_response = self.embedding_layer(response)

        # (batch_size, batch_sentence_len, dim_gru1)
        vec_utt_response, _ = self.gru1_layer(vec_word_response)

        vec_word_response = vec_word_response.permute(0, 2, 1)
        vec_utt_response = vec_utt_response.permute(0, 2, 1)

        list_vec_match = []
        for vec_word_history in vec_word_list_history:
            vec_utt_history, _ = self.gru1_layer(vec_word_history)
            # (batch_size, batch_sentence_len, dim_embedding)
            # (batch_size, batch_sentence_len, dim_embedding)
            matrix1 = torch.matmul(vec_word_history, vec_word_response)
            # assert(matrix1.shape == (batch_size, batch_sentence_len, batch_sentence_len))

            matrix2 = torch.einsum('aij,jk->aik', vec_utt_history, self.linear_transfom_op)
            matrix2 = torch.matmul(matrix2, vec_utt_response)

            # (batch_size,channel,batch_sentence_len,batch_sentence_len)
            matrix = torch.stack([matrix1, matrix2], dim=1)
            cnn_matrix = torch.nn.functional.relu(self.conv1_layer(matrix))
            cnn_matrix = self.pool_layer(cnn_matrix)

            vec_cnn = cnn_matrix.view(cnn_matrix.size(0), -1)
            vec_match = torch.nn.functional.tanh(self.cnn_linear_layer(vec_cnn))
            list_vec_match.append(vec_match)

        # (batch_size, n_history, dim_sim)
        _, last_hidden = self.gru2_layer(torch.stack(list_vec_match, dim=1))

        # h_n of shape (num_layers * num_directions, batch, hidden_size)
        last_hidden = torch.squeeze(last_hidden)
        logits = self.gru_linear_layer(last_hidden)

        ## when use CrossEntropyLoss, it would call softmax function
        ## or just NLLLoss
        # y_pred = torch.nn.functional.softmax(logits)
        y_pred = logits
        return y_pred


def train_epoch(model, loss_fn, optimizer, dataloader, use_gpu):
    model.train()

    loss_train = 0
    cnt_acc = 0
    for i, data in enumerate(dataloader):
        # 将数据从 train_loader 中读出来,一次读取的样本数是batch_size个

        inputs, labels = data
        inputs = inputs.permute(1, 0, 2)
        list_utterance = inputs[:-1]
        response = inputs[-1]
        # logging.debug(list_utterance.size())
        # logging.debug(response.size())

        if use_gpu:
            list_utterance = list_utterance.cuda()
            response = response.cuda()
            labels = labels.cuda()

        y_pred = model(list_utterance, response)
        loss = loss_fn(y_pred, labels)
        loss_train += loss.item()
        y_label = torch.argmax(y_pred, dim = 1)
        cnt_acc += torch.sum(labels.data == y_label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 200 == 199:
            logging.info('step %5d loss_train: %.3f' % (i + 1, loss_train / 200.0))
            logging.info('step %5d acc_train: %.3f' % (i + 1, cnt_acc / 200.0 / batch_size))
            loss_train = 0.0
            cnt_acc = 0

def eval_epoch(model, loss_fn, optimizer, dataloader, use_gpu):
    model.eval()

    with torch.no_grad():
        loss_eval = 0
        cnt_acc = 0
        total_loss = 0
        total_cnt = 0
        for i, data in enumerate(dataloader):
            # 将数据从 train_loader 中读出来,一次读取的样本数是batch_size个

            inputs, labels = data
            inputs = inputs.permute(1, 0, 2)
            list_utterance = inputs[:-1]
            response = inputs[-1]

            if use_gpu:
                list_utterance = list_utterance.cuda()
                response = response.cuda()
                labels = labels.cuda()

            y_pred = model(list_utterance, response)
            loss = loss_fn(y_pred, labels)
            loss_eval += loss.item()
            total_loss += loss.item()

            y_label = torch.argmax(y_pred, dim = 1)
            cnt_acc += torch.sum(labels.data == y_label)
            total_cnt += labels.size()[0]

            if i % 20 == 19:
                logging.info('step %5d loss_eval: %.3f' % (i + 1, loss_eval / 20.0))
                logging.info('step %5d acc_eval: %.3f' % (i + 1, cnt_acc / 20.0 / batch_size))

                loss_eval = 0.0
                cnt_acc = 0

    return total_loss/total_cnt


def predict(model, dataloader, use_gpu):
    model.eval()

    list_pred = []
    list_label = []

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            # 将数据从 train_loader 中读出来,一次读取的样本数是batch_size个
            inputs, labels = data
            inputs = inputs.permute(1, 0, 2)
            list_utterance = inputs[:-1]
            response = inputs[-1]

            if use_gpu:
                list_utterance = list_utterance.cuda()
                response = response.cuda()
                labels = labels.cuda()

            y_pred = model(list_utterance, response).cpu()

            for pred in y_pred:
                label = torch.argmax(pred)
                list_label.append(label.numpy())
                list_pred.append(torch.nn.functional.softmax(pred).numpy())

    return np.array(list_label), np.array(list_pred)



def get_SMN_dataloader(filename, dict_word2id):
    data_train = pd.read_csv(filename, header=None, sep="\t")
    logging.info(filename + "\n" + str(data_train.iloc[0:10]))
    y_train = np.array(data_train.pop(data_train.shape[1] - 1))
    logging.info("label: " + str(y_train[0:10]))

    x_train = np.zeros((data_train.shape[0], data_train.shape[1], max_len_utterance))
    for i in range(data_train.shape[0]):
        for j in range(data_train.shape[1]):
            sentence = data_train.iat[i, j]
            sentence_cuted = list(jieba.cut(str(sentence)))
            x_train[i][j][0] = dict_word2id["[BOS]"]
            cnt = 1
            for word in sentence_cuted:
                if cnt >= max_len_utterance - 1:
                    break
                if word in dict_word2id:
                    x_train[i][j][cnt] = dict_word2id[word]
                else:
                    x_train[i][j][cnt] = dict_word2id["[OOV]"]
                cnt += 1
            while (cnt < max_len_utterance - 1):
                x_train[i][j][cnt] = dict_word2id["[PAD]"]
                cnt += 1
            x_train[i][j][cnt] = dict_word2id["[EOS]"]

    dataset_train = torch.utils.data.TensorDataset(torch.from_numpy(x_train).long(), torch.from_numpy(y_train).long())
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    return dataloader_train

def run_SMN(use_gpu = True):
    embedding_loader = JDEmbeddingLoader()
    data_embedding, dict_word2id = embedding_loader.get_embedding()

    logging.info("creating model and loading data...")
    model = SMNModel(torch.from_numpy(np.array(data_embedding)), use_gpu)
    logging.info(str(model))
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    if use_gpu:
        model = model.cuda()
        loss_fn = loss_fn.cuda()

    dataloader_train = get_SMN_dataloader("./data/train.tsv", dict_word2id)
    dataloader_dev = get_SMN_dataloader("./data/dev.tsv", dict_word2id)
    dataloader_predict = get_SMN_dataloader("./data/test.tsv", dict_word2id)

    patience_max = 5
    best_loss = 10000
    patience_count = 0
    for epoch in range(100):
        logging.info("="*48)
        logging.info("epoch " + str(epoch+1))
        logging.info("="*48)

        train_epoch(model, loss_fn, optimizer, dataloader_train, use_gpu)
        eval_loss = eval_epoch(model, loss_fn, optimizer, dataloader_dev, use_gpu)

        if eval_loss < best_loss:
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience_max:
                print('-> Early stopping at epoch {}...'.format(epoch))
                break


    x_label, x_pred = predict(model, dataloader_predict, use_gpu)
    logging.info(x_label)
    data_pred = pd.DataFrame(x_pred)
    data_pred.to_csv("./output/test_results.tsv", sep="\t", header = None, index = False)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)
    logging.getLogger("jieba").setLevel(logging.INFO)
    run_SMN()