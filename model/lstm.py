import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable as Var

from model.utils import *


class LSTM(nn.Module):
    def __init__(self, in_dim, out_dim, p_dropout=0.0):
        super().__init__()

        self.input_dim = in_dim
        self.output_dim = out_dim
        self.dropout = p_dropout

        self.W_i = nn.Linear(in_dim, out_dim)
        init.xavier_uniform(self.W_i.weight)
        self.W_i.bias = nn.Parameter(torch.FloatTensor(out_dim).zero_())
        self.U_i = nn.Linear(out_dim, out_dim, bias=False)
        init.orthogonal(self.U_i.weight)

        self.W_f = nn.Linear(in_dim, out_dim)
        init.xavier_uniform(self.W_f.weight)
        self.W_f.bias = nn.Parameter(torch.FloatTensor(out_dim).fill_(1.0))
        self.U_f = nn.Linear(out_dim, out_dim)
        init.orthogonal(self.U_f.weight)

        self.W_c = nn.Linear(in_dim, out_dim)
        init.xavier_uniform(self.W_c.weight)
        self.W_c.bias = nn.Parameter(torch.FloatTensor(out_dim).fill_(0.0))
        self.U_c = nn.Linear(out_dim, out_dim)
        init.orthogonal(self.U_c.weight)

        self.W_o = nn.Linear(in_dim, out_dim)
        init.xavier_uniform(self.W_o.weight)
        self.W_o.bias = nn.Parameter(torch.FloatTensor(out_dim).fill_(0.0))
        self.U_o = nn.Linear(out_dim, out_dim)
        init.orthogonal(self.U_o.weight)

    def forward_node(self, xi, xf, xo, xc, h_prev, c_prev, dr_H):
        i = F.sigmoid(xi + self.U_i(h_prev * dr_H[0]))
        f = F.sigmoid(xi + self.U_f(h_prev * dr_H[1]))
        c = f * c_prev + i * F.tanh(xc + self.U_c(h_prev * dr_H[2]))
        o = F.sigmoid(xo + self.U_o(h_prev * dr_H[3]))
        h = o * F.tanh(c)
        return h, c

    def forward(self, X):
        batch_size = X.size()[0]
        length = X.size()[1]
        # (4, batch_size, input_dim)
        dr_X = dropout_matrix(4, batch_size, 1, self.input_dim, train=self.training, cuda=X.is_cuda, p=self.dropout)
        # (4, batch_size, output_dim)
        dr_H = dropout_matrix(4, batch_size, self.output_dim, train=self.training, cuda=X.is_cuda, p=self.dropout)

        Xi = self.W_i(X * dr_X[0])
        Xf = self.W_f(X * dr_X[1])
        Xc = self.W_c(X * dr_X[2])
        Xo = self.W_o(X * dr_X[3])

        h = init_var(batch_size, self.output_dim, scale=0.1, cuda=X.is_cuda, training=self.training)
        c = init_var(batch_size, self.output_dim, scale=0.1, cuda=X.is_cuda, training=self.training)
        h_hist = []
        c_hist = []
        for t in range(length):
            xi, xf, xo, xc = Xi[:, t, :].squeeze(1), \
                             Xf[:, t, :].squeeze(1), \
                             Xo[:, t, :].squeeze(1), \
                             Xc[:, t, :].squeeze(1)

            h, c = self.forward_node(xi, xf, xo, xc,
                                     h, c, dr_H)
            h_hist.append(h)
            c_hist.append(c)

        return torch.stack(h_hist, dim=1), torch.stack(c_hist, dim=1)