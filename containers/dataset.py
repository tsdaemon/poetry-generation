from copy import deepcopy
import torch.utils.data as data
import torch


class Dataset(data.Dataset):
    def __init__(self, file_name, vocab):
        super(Dataset, self).__init__()
        self.vocab = vocab

        self.vectors = list(self.load(file_name))

        self.size = len(self.vectors)

        self.torch_vectors = self.prepare_torch(self.vectors)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return deepcopy(self.torch_vectors[index])

    def get_batch(self, indices):
        return deepcopy(self.torch_vectors[indices])

    def load(self, file_name):
        with open(file_name, 'r') as f:
            yield list(map(int, f.readline().split(' ')))

    def prepare_torch(self, vectors):
        return torch.LongTensor(vectors)



