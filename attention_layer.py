class BilinearSeqAttn(nn.Module):
    """A bilinear attention layer over a sequence X w.r.t y:
    * o_i = x_i'Wy for x_i in X.
    """
    def __init__(self, x_size, y_size, opt, identity=False):
        super(BilinearSeqAttn, self).__init__()
        if not identity:
            self.linear = nn.Linear(y_size, x_size)
        else:
            self.linear = None

    def forward(self, x, y, x_mask):
        """
        x = [batch, len, h1]
        y = [batch, h2]
        x_mask = [batch, len]
        """
        x = dropout(x, p=my_dropout_p, training=self.training)
        y = dropout(y, p=my_dropout_p, training=self.training)

        Wy = self.linear(y) if self.linear is not None else y
        xWy = x.bmm(Wy.unsqueeze(2)).squeeze(2)
        xWy.data.masked_fill_(x_mask.data, -float('inf'))
        return xWy # [batch,len]

# bmm: batch matrix multiplication
# unsqueeze: add singleton dimension
# squeeze: remove singleton dimension
def weighted_avg(x, weights): 
    """ x = [batch, len, d]
        weights = [batch, len]
    """
    return weights.unsqueeze(1).bmm(x).squeeze(1)

if __name__ == "__main__":
    # [batch,sentence_len,hidden_dim], [batch,hidden_dim2] -> [batch,sentence_len]
    sentence_weights = bilinear_seq_attn(sentence_hiddens, y, sentence_mask) 

    # [batch,hidden_dim]
    sentence_avg_hidden = weighted_avg(sentence_hiddens, sentence_weights)