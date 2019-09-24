import torch
import torch as T
from dnc.dnc import DNC

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

import os

from tqdm import tqdm

from utils import generate_data, generate_result_images, compute_cost

def execute(rnn, x, y):
    # Execute the model
    (chx, mhx, rv) = (None, None, None)
    output, (chx, mhx, rv), v = rnn(x, (None, mhx, None), reset_experience=True, pass_through_memory=True)

    # Get only the final part of the sequence
    y_out = sigm(output[:, :-sequence_length, :-3])
    y = y[:,:,:-3]

    return compute_cost(sigm(output[:, -sequence_length:, :-3]), y, batch_size=1).item()

# Set the arg parser
parser = argparse.ArgumentParser(description='PyTorch DNC Priority Sort Task')
parser.add_argument('--model', type=str, default="./model.pt", help='Model checkpoint used to perform the tests.')
parser.add_argument('--iterations', type=int, default=1000, help="Number of tests which will be performed.")
parser.add_argument('--seed', type=int, default=42, help="Seed used for the random number generator.")

# Parse the arguments
args = parser.parse_args()

# Set the seed
np.random.seed(int(args.seed))

# Get the information from the model. The configs are the following:
configs = os.path.basename(args.model).split("_")
sequence_num_of_bits = int(configs[3])
rnn_type = str(configs[4])
nhid = int(configs[5])
memory_type = str(configs[6])
steps = int(configs[7])
batch_size = int(configs[8])
mem_size = int(configs[9])
mem_slot = int(configs[10])
sequence_length = int(configs[11])
iterations = int(configs[12])
non_uniform_priority = bool(configs[13].split(".")[0])

# Generate the model
rnn = DNC(input_size=sequence_num_of_bits+3,
        hidden_size=nhid,
        rnn_type=rnn_type,
        num_layers=1,
        num_hidden_layers=2,
        dropout=0,
        nr_cells=mem_slot,
        cell_size=mem_size,
        read_heads=5,
        gpu_id=-1,
        debug=True,
        batch_first=True,
        independent_linears=False)
rnn.load_state_dict(torch.load(args.model))

# Total costs
total_costs = pd.DataFrame(columns=["non_uniform_not_ordered",
"non_uniform_ordered",
"uniform_not_ordered",
"uniform_ordered",
"uniform_ordered_mixture",
"uniform_not_ordered_mixture"])

# Execute the evaluation
sigm = T.nn.Sigmoid()

for i in tqdm(range(0, args.iterations)):

    x, y = generate_data(1, sequence_length, sequence_num_of_bits+3, steps=steps, non_uniform=True, ordered=False)
    a = execute(rnn, x, y)

    x, y = generate_data(1, sequence_length, sequence_num_of_bits+3, steps=steps, non_uniform=True, ordered=True)
    b = execute(rnn, x, y)

    x, y = generate_data(1, sequence_length, sequence_num_of_bits+3, steps=steps, non_uniform=False, ordered=False)
    c = execute(rnn, x, y)

    x, y = generate_data(1, sequence_length, sequence_num_of_bits+3, steps=steps, non_uniform=False, ordered=True)
    d = execute(rnn, x, y)

    x, y = generate_data(1, sequence_length, sequence_num_of_bits+3, steps=steps, non_uniform=False, ordered=True, mixture=True)
    e = execute(rnn, x, y)

    x, y = generate_data(1, sequence_length, sequence_num_of_bits+3, steps=steps, non_uniform=False, ordered=False, mixture=True)
    f = execute(rnn, x, y)

    total_costs=total_costs.append(
        {"non_uniform_not_ordered": a,
        "non_uniform_ordered": b,
        "uniform_not_ordered": c,
        "uniform_ordered": d,
        "uniform_not_ordered_mixture": e,
        "uniform_ordered_mixture": f}, ignore_index=True)

ax = sns.boxplot(data=total_costs, orient="v")
plt.show()
