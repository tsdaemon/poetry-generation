import numpy as np
from numpy.random import RandomState
import random


class SoftmaxGenerator:
    def __init__(self, model, start_idx, end_idx, vocab_size, max_time_step):
        self.model = model
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.max_time_step = max_time_step
        self.vocab_size = vocab_size

    def generate(self, random_seed=None):
        curr_input = [self.start_idx]
        curr_output = None
        h_init = None
        c_init = None
        rng = RandomState(random_seed)
        result = []
        step = 0

        while curr_output != self.end_idx and step < self.max_time_step:
            distribution, h_init, c_init = self.model(curr_input, h_init, c_init)

            # transfrom to cumulative distribution and move to numpy array
            cum_distribution = np.cumsum(distribution)
            rnd_number = rng.uniform(0, 1.0)
            curr_output = np.argmax(rnd_number < cum_distribution)

            result.append(curr_output)
            curr_input = [curr_output]
            step += 1

        return result

    def generate_q(self, epsilon):
        curr_input = [self.start_idx]
        curr_output = None
        c_init = None
        h_init = None
        results = []
        step = 0
        while curr_output != self.end_idx and step < self.max_time_step:
            q, distribution, h_init, c_init = self.model.forward_q(curr_input, h_init, c_init)

            # no fucking epsilon
            # random_char = random.random() < epsilon
            #
            # if random_char:
            #     curr_output = np.random.randint(0, self.vocab_size)
            # else:

            cum_distribution = np.cumsum(distribution)
            rnd_number = np.random.uniform(0, 1.0)
            curr_output = np.argmax(rnd_number < cum_distribution)

            results.append([curr_input[0], curr_output])
            curr_input = [curr_output]
            step += 1

        return results

