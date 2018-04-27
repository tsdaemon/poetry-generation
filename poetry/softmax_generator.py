import torch.autograd.variable as Var
from numpy.random import RandomState


class SoftmaxGenerator():
    def __init__(self, model, start_symbol, end_symbol):
        self.model = model
        self.start_symbol = start_symbol
        self.end_symbol = end_symbol

    def generate(self, random_seed = None):
        curr_input = [self.start_symbol]
        curr_output = None
        rng = RandomState(random_seed)
        result = []

        while curr_output != self.end_symbol:
            distribution = self.model(Var(curr_input, requires_grad=False))

            # transfrom to cumulative distribution and move to numpy array
            cum_distribution = distribution.cumsum(0).data.numpy()
            rnd_number = rng.uniform(0, 1.0)
            for i, cum_proba in enumerate(cum_distribution):
                if rnd_number < cum_proba:
                    curr_output = i
                    break
            result.append(curr_output)
            curr_input = [curr_output]

        return result
