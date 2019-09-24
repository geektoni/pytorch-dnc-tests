import numpy as np

class PriorityGenerator:

    def __init__(self, batch_size, length, seed=42):

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

        self.batch_size = batch_size
        self.length = length
        self.distr = [-1, -1, -1]

    def sample(self):

        # Sample how many distributions we will use
        mixture_size = np.random.uniform(1, 5)

        # Get indexes of which distribusion we will use
        mixture_indexes = np.random.choice(5, int(mixture_size))

        # Get multinomial probabilities and normalize them such to sum to 1
        multinomial_prob = np.random.uniform(1, 0.3, int(mixture_size))
        multinomial_prob /= np.sum(multinomial_prob)

        # Sample from all the betas
        self.sample_from_all()

        # Choose which one of the distribution we will use
        choice =  np.random.multinomial(1, multinomial_prob)

        # Return the choose priority
        final_index = mixture_indexes[np.where(choice==1)[0]][0]
        return self.distr[final_index]

    def sample_from_all(self):

        a, b = self.sample_alpha_beta((0.8, 1), (2,10))
        a2, b2 = self.sample_alpha_beta((2, 10), (0.8,1))
        a3, b3 = self.sample_alpha_beta((2, 10), (2,10))
        a4, b4 = self.sample_alpha_beta((2, 3), (5,8))
        a5, b5 = self.sample_alpha_beta((5, 8), (2,3))

        self.distr = [
            np.random.beta(a, b, (self.batch_size, self.length, 1)),
            np.random.beta(a2, b2, (self.batch_size, self.length, 1)),
            np.random.beta(a3, b3, (self.batch_size, self.length, 1)),
            np.random.beta(a4, b4, (self.batch_size, self.length, 1)),
            np.random.beta(a5, b5, (self.batch_size, self.length, 1)),
        ]

    def sample_alpha_beta(self, range_a, range_b):
        alpha = np.random.uniform(range_a[0], range_a[1], 1)
        beta = np.random.uniform(range_b[0], range_b[1], 1)
        return alpha, beta
