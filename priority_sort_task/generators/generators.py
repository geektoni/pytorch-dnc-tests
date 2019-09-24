import numpy as np

class PriorityGenerator:

    def __init__(self, batch_size, length, seed=42):
        np.random.seed(seed)

        self.batch_size = batch_size
        self.length = length
        self.mm = MixturePriorityModel(batch_size, length, seed)

    def sample(self, non_uniform=False, mixture=False):
        if non_uniform:
            alpha = np.random.uniform(1, 100)
            beta = np.random.uniform(1, 100)
            priority = np.random.beta(alpha, beta, (self.batch_size, self.length, 1))
        elif mixture:
            priority = self.mm.sample()
        else:
            priority = np.random.uniform(-1, 1, (self.batch_size, self.length, 1))
        return priority


class MixturePriorityModel:

    def __init__(self, batch_size, length, seed=42):
        np.random.seed(seed)

        self.batch_size = batch_size
        self.length = length
        self.distr = [-1, -1, -1]

    def sample(self):
        self.sample_from_all()
        choice =  np.random.multinomial(1, [1.0/3.0, 1.0/3.0, 1.0/3.0])
        return self.distr[np.where(choice==1)[0][0]]

    def sample_from_all(self):
        self.Uniform = np.random.uniform(-1, 1, (self.batch_size, self.length, 1))
        self.Normal = np.random.beta(5, 1, (self.batch_size, self.length, 1))
        self.Beta = np.random.beta(5, 1, (self.batch_size, self.length, 1))
        self.distr = [self.Uniform, self.Normal, self.Beta]
