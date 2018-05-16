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
            for i, cum_proba in enumerate(cum_distribution):
                if rnd_number < cum_proba:
                    curr_output = i
                    break
            result.append(curr_output)
            curr_input = [curr_output]
            step += 1

        return result

    def generate_q(self, epsilon):
        curr_input = [self.start_idx]
        curr_idx = None
        results = []
        step = 0
        while curr_idx != self.end_idx and step < self.max_time_step:
            curr_output, distribution = self.model.forward_q(curr_input)

            random_char = random.random() < epsilon
            if random_char:
                curr_idx = random.randint(self.vocab_size)
            else:
                cum_distribution = np.cumsum(distribution)
                rnd_number = np.random.uniform(0, 1.0)
                for i, cum_proba in enumerate(cum_distribution):
                    if rnd_number < cum_proba:
                        curr_idx = i
                        break

            results.append(curr_input, curr_output, curr_idx)
            curr_input = [curr_idx]
            step += 1

        return results

