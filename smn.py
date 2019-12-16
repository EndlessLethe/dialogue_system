import torch
import numpy as np

n_dict =
dim_embedding = 300 # 200 dim trained with word2vec in passage, here use char-
dim_gru1 = 600 # 200 in passage
dim_gru2 = 100 # 50 in passage
n_in_channel = 2 # word and segment matrix
n_feature_map = 8 # 8 in passage
kernel_shape = (3, 3) # (3, 3) in passage
magic_3 = 3 # use to control cnn arg and output shape
pool_shape = (magic_3, magic_3) # (3, 3) in passage
pool_stride = (magic_3, magic_3)
gru_dorpout = 0.3
dim_sim_vec = 50


learning_rate = 1e-4
max_len_utterance = 50
n_utterance = 7
n_label = 2


class SMNModel(torch.nn.Module):
    """
    This model needs JD-pretrained-char-embedding

    For each sample:
    Input - list_history and query
    Output -

    Expected final output:
        shape = (n_sample, n_label) n_label = 0 (not related) or 1(related)
    """
    def __init__(self):
        super(SMNModel, self).__init__()

        ## define layers and then init weights

        # whether to set padding_idx?
        self.embedding_layer = torch.nn.Embedding(num_embeddings=n_dict, embedding_dim=dim_embedding, padding_idx=None)
        self.gru1_layer = torch.nn.GRU(input_size=dim_embedding, hidden_size=dim_gru1, batch_first=True,
                                      dropout=gru_dorpout, bidirectional=True)

        self.conv1_layer = torch.nn.Conv2d(in_channels=n_in_channel, out_channels=n_feature_map, kernel_size=kernel_shape)
        self.relu_layer = torch.nn.ReLU()
        self.pool_layer = torch.nn.MaxPool2d(kernel_size=pool_shape, stride=pool_stride)
        ## only one-layer cnn
        # self.conv2_layer = torch.nn.Conv2d(in_channels=n_feature_map, out_channels=n_feature_map, kernel_size=kernel_shape)

        # flatten all matrices of h'u and h'r as vi
        self.cnn_linear_layer = torch.nn.Linear(in_features=int(dim_gru1 / magic_3 * n_feature_map), out_features=dim_sim_vec)
        self.gru2_layer = torch.nn.GRU(input_size=dim_sim_vec, hidden_size=dim_gru2, batch_first=True,
                                       dropout=gru_dorpout, bidirectional=True)

        # input the concat of all h'u and h'r, output matching socre
        self.gru_linear_layer = torch.nn.Linear(in_features=dim_gru2*n_utterance, out_features=n_label)


        # load embedding from file

        ## build model



        def forword(self, list_history, response):
