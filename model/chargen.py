import torch
import torch.nn as nn
import torch.nn.init as init
import torch.autograd.variable as Var

import Constants
from model.lstm import LSTM
from model.general import LogSoftmaxDense


class CharGen(nn.Module):
    def __init__(self, config, vocab_length):
        super().__init__()

        self.embed = nn.Embedding(len(vocab_length), config.char_embed_dim, padding_idx=Constants.PAD)
        init.normal(self.word_embedding.weight, 0.0, 0.2)
        self.word_embedding.weight.requires_grad = not config.freeze_embeds

        self.lstm = LSTM(input_dim=config.char_embed_dim, out_dim=config.hidden_dim, p_dropout=config.dropout)

        self.logsoftmax_dense = LogSoftmaxDense(config.hidden_dim, len(vocab_length))

        self.is_cuda = config.cuda

    def forward(self, X):  # X: (1)
        # (1, char_embed_dim, 1)
        X = self.embed(X).unsqueeze(0)

        # (1, hidden_dim, 1)
        X = self.lstm(X)

        # (vocab_length)
        return self.logsoftmax_dense(X).squeeze(0).squeeze(1)

    def forward_train(self, X, y):
        # (batch_num, char_embed_dim, max_poem_length)
        X = self.embed(X)

        # (batch_num, hidden_dim, max_poem_length)
        X = self.lstm(X)

        # (batch_num, vocab_length, max_poem_length)
        loglikelihoods = self.logsoftmax_dense.forward_train(X)

        # (batch_num, max_poem_length)
        loglikelihoods = loglikelihoods.gather(2, Var(y, requires_grad=False)).squeeze(2)

        loss = torch.neg(torch.sum(loglikelihoods))

        return loss
