import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable as Var

import Constants
from model.lstm import LSTM
from model.general import LogSoftmaxDense
from model.utils import from_list


class CharGen(nn.Module):
    def __init__(self, config, vocab_length):
        super().__init__()

        self.embed = nn.Embedding(vocab_length, config.char_embed_dim, padding_idx=Constants.PAD)
        init.normal_(self.embed.weight, 0.0, 0.2)
        self.embed.weight.requires_grad = not config.freeze_embeds

        self.lstm1 = LSTM(in_dim=config.char_embed_dim, out_dim=config.hidden_dim,
                         p_dropout=config.dropout)

        self.lstm2 = LSTM(in_dim=config.hidden_dim, out_dim=config.hidden_dim,
                           p_dropout=config.dropout)

        self.logsoftmax_dense = LogSoftmaxDense(config.hidden_dim, vocab_length)

        self.is_cuda = config.cuda
        self.layers = config.layers

    def forward(self, X):  # X: (1)
        # convert from usual vector
        X = Var(from_list(X, self.is_cuda, torch.LongTensor), requires_grad=False)
        # (1,  1, char_embed_dim)
        X = self.embed(X).unsqueeze(0)

        # (1, 1, hidden_dim)
        h, c = self.lstm1(X)

        # (1, 1, hidden_dim)
        if self.layers > 1:
            h, c = self.lstm2(h)

        # (hidden_dim)
        h = h.squeeze(0).squeeze(0)

        # (vocab_length)
        dist = self.logsoftmax_dense(h)

        # convert back to numpy
        if self.is_cuda:
            dist = dist.cpu()

        return dist.data.numpy()

    def forward_train(self, X, y):
        # (batch_num, max_poem_length, char_embed_dim)
        X = self.embed(X)

        # (batch_num, max_poem_length, hidden_dim)
        h, c = self.lstm1(X)

        # (batch_num, max_poem_length, hidden_dim)
        if self.layers > 1:
            h, c = self.lstm2(h)

        # (batch_num, max_poem_length, vocab_length)
        loglikelihoods = self.logsoftmax_dense.forward_train(h)

        assert (loglikelihoods <= 0).all()

        # (batch_num, max_poem_length)
        loglikelihoods = loglikelihoods.gather(2, y.unsqueeze(2)).squeeze(2)

        loss = torch.neg(torch.sum(loglikelihoods))

        assert (loss > 0).all(), "NLL can not be less than zero"

        return loss
