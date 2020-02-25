import torch
import torch as T
from dnc.dnc import DNC

import numpy as np
import pandas as pd
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
import argparse

import os
import glob

from tqdm import tqdm

from utils import generate_data, generate_result_images, compute_cost

def execute(rnn, x, y, sequence_length):
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
parser.add_argument('--iterations', type=int, default=100, help="Number of tests which will be performed.")
parser.add_argument('--seed', type=int, default=42, help="Seed used for the random number generator.")

# Parse the arguments
args = parser.parse_args()

# Set the seed
np.random.seed(int(args.seed))

# Total costs
total_costs = pd.DataFrame(columns=["Network / Uniform Sampling", "sampling_type", "cost"])

values = pd.DataFrame(columns=["Network", "Not Uniform", "length", "cost"])

all_models = glob.glob(args.model+"/*")

for current_model in all_models:

    print("Analyzing ", current_model)

    # Get the information from the model. The configs are the following:
    configs = os.path.basename(current_model).split("_")
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
    non_uniform_priority = configs[13].lower() == True
    mixin = configs[14].lower() == "true"
    copy_mode = False

    # Generate the model
    rnn = DNC(input_size=sequence_num_of_bits+3,
            hidden_size=nhid,
            rnn_type=rnn_type,
            num_layers=1,
            num_hidden_layers=2,
            dropout=0,
            nr_cells=mem_slot,
            cell_size=mem_size,
            read_heads=1,
            gpu_id=-1,
            debug=True,
            batch_first=True,
            independent_linears=False,
            copy_mode=copy_mode)

    rnn.load_state_dict(torch.load(current_model))

    # Execute the evaluation
    sigm = T.nn.Sigmoid()

    sequence_length -= 1

    for i in tqdm(range(0, args.iterations)):

        x, y, _ = generate_data(1, sequence_length, sequence_num_of_bits+3, steps=steps, non_uniform=True, ordered=False)
        a = execute(rnn, x, y, sequence_length)

        x, y, _ = generate_data(1, sequence_length, sequence_num_of_bits+3, steps=steps, non_uniform=True, ordered=True)
        b = execute(rnn, x, y, sequence_length)

        x, y, _ = generate_data(1, sequence_length, sequence_num_of_bits+3, steps=steps, non_uniform=False, ordered=False)
        c = execute(rnn, x, y, sequence_length)

        x, y, _ = generate_data(1, sequence_length, sequence_num_of_bits+3, steps=steps, non_uniform=False, ordered=True)
        d = execute(rnn, x, y, sequence_length)

        x, y, _ = generate_data(1, sequence_length, sequence_num_of_bits+3, steps=steps, non_uniform=False, ordered=True, mixture=True)
        e = execute(rnn, x, y, sequence_length)

        x, y, _ = generate_data(1, sequence_length, sequence_num_of_bits+3, steps=steps, non_uniform=False, ordered=False, mixture=True)
        f = execute(rnn, x, y, sequence_length)

        partial_results = [a,b,c,d,e,f]
        #partial_results_title = ["non_uniform_not_ordered",
        #"non_uniform_ordered",
        #"uniform_not_ordered",
        #"uniform_ordered",
        #"uniform_not_ordered_mixture",
        #"uniform_ordered_mixture"
        #]
        partial_results_title = [0,1,2,3,4,5,6]

        for pres in zip(partial_results, partial_results_title):
            total_costs = total_costs.append(
            {
            "Network / Uniform Sampling": rnn_type+"/"+str(mixin),
            "sampling_type": pres[1],
            "cost": pres[0]
            }, ignore_index=True
            )

    max_length = 20
    for i in tqdm(range(0, max_length)):
        sequence_length_tmp = 2+i
        mean_result = 0
        for i in range(0, 50):
            x, y, _ = generate_data(1, sequence_length_tmp, sequence_num_of_bits+3, steps=steps, ordered=False, mixture=True)
            a = execute(rnn, x, y, sequence_length_tmp)
            mean_result = mean_result+a
            values = values.append({"Network": rnn_type,
            "Not Uniform": mixin,
            "length": sequence_length_tmp, "cost": a}, ignore_index=True)
    #values.append((mean_result/100.0)/sequence_length_tmp)
    #values.append((mean_result/100.0))

fig, axs = plt.subplots(2, figsize=(10, 6))

sns.lineplot(x="length", y="cost", err_style="bars", hue="Network", style="Not Uniform", ax=axs[0], data=values)
sns.boxplot(data=total_costs, y="cost", x="sampling_type", hue="Network / Uniform Sampling", orient="v", ax=axs[1])


axs[1].set_xticklabels(["n-unif", "n-unif-ord", "unif", "unif-ord", "unif-mix", "unif-mix-ord"])
axs[1].tick_params(labelrotation=90)
axs[1].set_xlabel("Sampling Strategy")

axs[0].set_xlabel("List Length")
axs[0].set_ylabel("Cost (errors)")
axs[0].set_xlim(1, max_length+3)

box = axs[1].get_position()
axs[1].set_position([box.x0, box.y0, box.width * 0.8, box.height])
axs[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Network / Uniform Sampling", fontsize=8)

box = axs[0].get_position()
axs[0].set_position([box.x0, box.y0, box.width * 0.8, box.height])
axs[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.tight_layout()
plt.savefig("DNC_evaluation.png", dpi=250, bbox_inches="tight")
