from copy import deepcopy
import torch.utils.data as data
import torch


class Dataset(data.Dataset):
    def __init__(self, file_name):
        super(Dataset, self).__init__()
        self.vectors = self.load(file_name)

        self.size = len(self.vectors)
        assert self.size > 10

        self.torch_vectors = self.prepare_torch(self.vectors)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        # (max_poem_length)
        poem = self.torch_vectors[index]
        X = poem[:-1]
        y = poem[1:]
        return X, y

    def get_batch(self, indices):
        # (batch_num, max_poem_length)
        poems = self.torch_vectors[indices]
        X = poems[:, :-1]
        y = poems[:, 1:]
        return X, y

    def load(self, file_name):
        with open(file_name, 'r') as f:
            return [[int(v) for v in line.split(' ')] for line in f.readlines()]

    def prepare_torch(self, vectors):
        return torch.LongTensor(vectors)

    def prepare_device(self, cuda):
        if cuda:
            self.torch_vectors = self.torch_vectors.cuda()



