import numpy as np
from numpy.random import RandomState


class SoftmaxGenerator:
    def __init__(self, model, start_idx, end_idx):
        self.model = model
        self.start_idx = start_idx
        self.end_idx = end_idx

    def generate(self, random_seed=None):
        curr_input = [self.start_idx]
        curr_output = None
        rng = RandomState(random_seed)
        result = []

        while curr_output != self.end_idx:
            distribution = self.model(curr_input)

            # transfrom to cumulative distribution and move to numpy array
            cum_distribution = np.cumsum(distribution)
            rnd_number = rng.uniform(0, 1.0)
            for i, cum_proba in enumerate(cum_distribution):
                if rnd_number < cum_proba:
                    curr_output = i
                    break
            result.append(curr_output)
            curr_input = [curr_output]

        return result
