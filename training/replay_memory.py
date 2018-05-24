import random

import Constants


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = args
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        sample = random.sample(self.memory, batch_size)
        # seq_size = max([len(input) for r, input, output in sample])
        #
        # # add padding to have equal lengths in a sample
        # sample_new = []
        # for r, input, output in sample:
        #     input += [Constants.PAD]*(seq_size - len(input))
        #     assert len(input) == seq_size
        #     output += [Constants.PAD] * (seq_size - len(output))
        #     assert len(output) == seq_size
        #     sample_new.append((r, input, output))

        return sample

    def __len__(self):
        return len(self.memory)