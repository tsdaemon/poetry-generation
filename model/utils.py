import torch
from torch.autograd import Variable as Var
import torch.nn as nn


def parameter_init_zero(*dims):
    return nn.Parameter(torch.FloatTensor(*dims).zero_())


def dropout_matrix(*dims, p=0.2, train=True, cuda=False):
    assert p <= 1.0 and p >= 0.0, "Invalid probability: {}".format(p)
    prob = 1-p
    # all 0.8, ok for evaluation
    d = Var(torch.FloatTensor(*dims).fill_(prob))
    if train:
        # all 1 or 0
        d = d.bernoulli()
    if cuda:
        d = d.cuda()
    return d


def zeros_var(*shape, cuda=False):
    t = torch.FloatTensor(*shape).zero_()
    if cuda:
        t = t.cuda()
    return Var(t, requires_grad=False)


def normal_var(*shape, cuda=False, scale=1.0):
    t = torch.FloatTensor(*shape).normal_(0.0, scale)
    if cuda:
        t = t.cuda()
    return Var(t, requires_grad=False)


def init_var(*shape, cuda=False, scale=1.0, training=True):
    if training:
        return normal_var(*shape, cuda=cuda, scale=scale)
    else:
        return zeros_var(*shape, cuda=cuda)


def from_list(ls, cuda, tensor_class):
    tensor = tensor_class(ls)
    if cuda:
        tensor = tensor.cuda()
    return tensor


def device_map_location(cuda):
    if cuda:
        return lambda storage, loc: storage.cuda()
    else:
        return lambda storage, loc: storage


def from_list(ls, cuda, tensor):
    tensor = tensor(ls)
    if cuda:
        tensor = tensor.cuda()
    return tensor


def cudafication(tensor, cuda):
    if cuda:
        tensor = tensor.cuda()
    return tensor