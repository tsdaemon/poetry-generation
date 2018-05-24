import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable as Var
import torch.nn.functional as F

import Constants
from model.lstm import LSTM
from model.general import LogSoftmaxDense
from model.utils import from_list, zeros_var


class CharGen(nn.Module):
    def __init__(self, config, vocab_length):
        super().__init__()

        self.embed = nn.Embedding(vocab_length, config.char_embed_dim, padding_idx=Constants.PAD)
        init.normal_(self.embed.weight, 0.0, 0.2)
        self.embed.weight.requires_grad = not config.freeze_embeds

        self.lstm1 = LSTM(in_dim=config.char_embed_dim, out_dim=config.hidden_dim,
                         p_dropout=config.dropout)

        self.layers = config.layers

        if self.layers > 1:
            self.lstm2 = LSTM(in_dim=config.hidden_dim, out_dim=config.hidden_dim,
                               p_dropout=config.dropout)
        if self.layers > 2:
            self.lstm3 = LSTM(in_dim=config.hidden_dim, out_dim=config.hidden_dim,
                               p_dropout=config.dropout)

        self.logsoftmax_dense = LogSoftmaxDense(config.hidden_dim, vocab_length)

        self.q_dense = nn.Linear(vocab_length, vocab_length)
        self.softmax = nn.Softmax(dim=-1)

        self.is_cuda = config.cuda
        self.hidden_dim = config.hidden_dim

    def _forward(self, X, h_init=None, c_init=None): # X: (1), h_init: (lstm_num, hidden_dim)
        """
        Forward is used for generation step by step, therefore it accepts initial hidden states
        from previous states. It accepts lists instead of tensors and return numpy arrays.
        :param X:
        :param h_init:
        :param c_init:
        :return:
        """

        if h_init is None:
            h_init = [zeros_var(self.hidden_dim, cuda=self.is_cuda)]*self.layers
        if c_init is None:
            c_init = [zeros_var(self.hidden_dim, cuda=self.is_cuda)]*self.layers

        # convert from usual vector
        X = Var(from_list(X, self.is_cuda, torch.LongTensor), requires_grad=False)

        # (1,  1, char_embed_dim)
        X = self.embed(X).unsqueeze(0)

        h_ls = []
        c_ls = []

        # (1, 1, hidden_dim)
        h, c = self.lstm1(X, h_init=h_init[0].unsqueeze(0), c_init=c_init[0].unsqueeze(0))
        h_ls.append(h.squeeze(0).squeeze(0))
        c_ls.append(c.squeeze(0).squeeze(0))

        # (1, 1, hidden_dim)
        if self.layers > 1:
            h, c = self.lstm2(h, h_init=h_init[1].unsqueeze(0), c_init=c_init[1].unsqueeze(0))
            h_ls.append(h.squeeze(0).squeeze(0))
            c_ls.append(c.squeeze(0).squeeze(0))

        # (1, 1, hidden_dim)
        if self.layers > 2:
            h, c = self.lstm3(h, h_init=h_init[2].unsqueeze(0), c_init=c_init[2].unsqueeze(0))
            h_ls.append(h.squeeze(0).squeeze(0))
            c_ls.append(c.squeeze(0).squeeze(0))

        # (hidden_dim)
        return h.squeeze(0).squeeze(0), h_ls, c_ls

    def forward(self, X, h_init=None, c_init=None):
        # (hidden_dim)
        h, h_hist, c_hist = self._forward(X, h_init, c_init)

        # (vocab_length)
        dist = self.logsoftmax_dense(h)

        # convert back to numpy
        if self.is_cuda:
            dist = dist.cpu()

        return dist.data.numpy(), h_hist, c_hist

    def forward_q(self, X, h_init=None, c_init=None):  # X: (1)
        # (hidden_dim)
        h, h_hist, c_hist = self._forward(X, h_init, c_init)

        # (vocab_length)
        pre_q = self.logsoftmax_dense.forward_q(h)

        # (vocab_length)
        q = self.q_dense(pre_q)

        # convert back to numpy
        if self.is_cuda:
            q = q.cpu()

        return q.data.numpy(), self.softmax(q).data.numpy(), h_hist, c_hist

    def _forward_train(self, X):
        # (batch_num, max_poem_length, char_embed_dim)
        X = self.embed(X)

        # (batch_num, max_poem_length, hidden_dim)
        h, c = self.lstm1(X)

        # (batch_num, max_poem_length, hidden_dim)
        if self.layers > 1:
            h, c = self.lstm2(h)

        # (batch_num, max_poem_length, hidden_dim)
        if self.layers > 2:
            h, c = self.lstm3(h)

        return h

    def forward_train(self, X, y):
        h = self._forward_train(X)

        # (batch_num, max_poem_length, vocab_length)
        loglikelihoods = self.logsoftmax_dense.forward_train(h)

        assert (loglikelihoods <= 0).all()

        # (batch_num, max_poem_length)
        loglikelihoods = loglikelihoods.gather(2, y.unsqueeze(2)).squeeze(2)

        loss = torch.neg(torch.sum(loglikelihoods))

        assert (loss > 0).all(), "NLL can not be less than zero"

        return loss

    def forward_train_q(self, X, q, idx):
        # (batch_num, max_poem_length, vocab_length)
        h = self._forward_train(X)

        # (batch_num, max_poem_length, vocab_length)
        pre_q_tensor = self.logsoftmax_dense.forward_q(h)

        # (batch_num, max_poem_length, vocab_length)
        q_tensor = self.q_dense(pre_q_tensor)

        # collect q values for selected indexes
        # (batch_num, max_poem_length)
        q_hat = q_tensor.gather(2, idx.unsqueeze(2)).squeeze(2)

        # Compute Huber loss
        loss = F.smooth_l1_loss(q_hat, q)

        return loss
