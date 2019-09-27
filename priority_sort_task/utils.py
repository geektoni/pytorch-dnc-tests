import sys

import numpy as np

import torch
import torch as T
import torch.nn.functional as F

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from dnc.dnc import DNC

from generators.generators import PriorityGenerator

def llprint(message):
    sys.stdout.write(message)
    sys.stdout.flush()

def generate_data(batch_size, length, size, steps=0, cuda=-1, non_uniform=False, ordered=False, mixture=False):

    # Generate the binary sequences of length equal to size.
    # We leave 2 bits empty for the priority and the delimiter.
    # Moreover, we add an empty vector which will be used as signal
    # for printing the result
    seq = np.random.binomial(1, 0.5, (batch_size, length, size - 3))
    seq = torch.from_numpy(seq)

    # Add priority number (just a single one drawn from the uniform distribution
    # or from the beta distribution)
    pg = PriorityGenerator(batch_size, length)
    priority = pg.sample(non_uniform, mixture)

    # Generate already ordered priority
    if ordered:
        priority = -np.sort(-priority, axis=1)

    priority = torch.from_numpy(priority)

    # Generate the first tensor
    inp = torch.zeros(batch_size, (length + 1), size)
    inp[:, :(length), :(size - 3)] = seq
    inp[:, :(length), (size - 3):] = priority
    inp[:, :(length), (size - 2):] = torch.zeros(batch_size, length, 1) # end of sequence
    inp[:, :(length), (size - 1):] = torch.zeros(batch_size, length, 1) # output the result

    # If the length is just 1, then add the delimiter and the end of sequence
    if (steps==0):
        inp[:, length, size - 2] = 1 # set the end of sequence
        inp[:, length, size - 1] = 1
    else:
        # add delimiter vector
        inp[:, length, size - 2] = 1 # set the end of sequence

    # For each step, we add an empty space
    for s in range(0,steps+1):

        if steps==0:
            break;

        inp_tmp = torch.zeros(batch_size, (1), size)

        # If this is the last repetition then we set the bit to 1.
        if (s == steps):
            inp_tmp[:, 0, size-1] = 1

        # Concatenate the tensor to the previous one
        inp = torch.cat((inp, inp_tmp), 1)

    # We then add an empty section in which we need to store the result
    # only if the steps are greater than 1
    #if steps > 0:
    inp_tmp = torch.zeros(batch_size, (length), size)
    inp = torch.cat((inp, inp_tmp), 1)

    outp = inp.numpy()

    # Strip all the binary vectors into a list
    # and sort the list by looking at the last column
    # (which will contain the priority)
    temp_total = []
    for b in range(batch_size):
        temp = []
        for i in range(length):
            temp.append(outp[b][i])
        if not ordered:
            temp.sort(key=lambda x: x[size-3], reverse=True)  # Sort elements descending order
        temp_total.append(temp)

    # FIXME
    # Ugly hack to present the tensor structure as the one
    # required by the framework
    output_final = []
    for b in range(len(temp_total)):
        layer = []
        for i in range(len(temp_total[b])):
            layer.append(np.array(temp_total[b][i]))
        output_final.append(layer)

    # Convert everything to numpy and to a tensor
    outp = torch.from_numpy(np.array(output_final))

    # Add an empy line at the end to simulate the delimiter
    outp = torch.cat((outp, torch.zeros(batch_size, 1, size)),1)

    if cuda != -1:
        inp = inp.cuda()
        outp = outp.cuda()

    return inp.float(), outp.float()[:, :-1, :], priority

def criterion(predictions, targets):
    return T.mean(
      -1 * F.logsigmoid(predictions) * (targets) - T.log(1 - F.sigmoid(predictions) + 1e-9) * (1 - targets)
    )

def binarize_array(array):
  binar = lambda x: 0 if x < 0.5 else 1
  bfunc = np.vectorize(binar)
  return np.array(bfunc(array))

def binarize_matrix(matrix):
  result = []
  binar = lambda x: 0 if x < 0.5 else 1
  bfunc = np.vectorize(binar)
  for r in matrix:
    result.append(np.array(bfunc(r)))
  return np.array(result)

def compute_cost(output, target_out, batch_size=1):
    """
    Function used to compute the cost of the generated
    sequences. They must be passed through a Sigmoid before
    they can be used.
    :param output: the predicted sequence
    :param target_out: the target sequence
    :return: the cost
    """
    y_out_binarized = []
    for t in output:
        y_out_binarized.append((t>0.5).data.cpu().numpy())
    y_out_binarized = T.from_numpy(np.array(y_out_binarized))

    # Binarize the original output
    target_output = []
    for t in target_out:
        target_output.append((t>0.5).data.cpu().numpy())
    target_output = T.from_numpy(np.array(target_output))

    # The cost is the number of error bits per sequence
    cost = torch.sum(torch.abs(y_out_binarized.cpu().float() - target_output.cpu().float()))/batch_size

    return cost

def generate_result_images(prediction, target, image_dir, experiment_name, epoch, args, model_path):

    x, y, priority = generate_data(1, args.sequence_max_length, args.input_size+3, steps=args.steps, non_uniform=False)

    print(priority.detach().numpy())
    print(np.argsort(-priority.detach().numpy(), axis=1))

    rnn = DNC(
        input_size=args.input_size+3,
        hidden_size=args.nhid,
        rnn_type=args.rnn_type,
        num_layers=args.nlayer,
        num_hidden_layers=args.nhlayer,
        dropout=args.dropout,
        nr_cells=args.mem_slot,
        cell_size=args.mem_size,
        read_heads=args.read_heads,
        gpu_id=args.cuda,
        debug=True,
        batch_first=True,
        independent_linears=args.independent_linears)
    rnn.load_state_dict(torch.load(model_path))

    (chx, mhx, rv) = (None, None, None)
    output, (chx, mhx, rv), v = rnn(x, (None, mhx, None), reset_experience=True, pass_through_memory=True)

    # This is needed if we want to use make_eval_plot
    sigm = T.nn.Sigmoid()
    prediction = sigm(output[:, -args.sequence_max_length:, :-3]).detach().numpy()[0]
    target = y[:,:,:-3].detach().numpy()[0]

    fig = plt.figure(figsize=(5,5))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(212)

    ax1.set_title("Result")
    ax2.set_title("Target")
    ax3.set_title("Input")

    x = x.detach().numpy()[0]
    prediction = np.swapaxes(prediction, 0, 1)
    target = np.swapaxes(target, 0, 1)
    x = np.swapaxes(x, 0, 1)

    prediction_bin = []
    for t in prediction:
        prediction_bin.append((t>0.5))
    prediction = T.from_numpy(np.array(prediction_bin))

    sns.heatmap(prediction, ax=ax1, vmin=0, vmax=1, linewidths=.5, linecolor="black", cmap="Greys", cbar=True)
    sns.heatmap(target, ax=ax2, vmin=0, vmax=1, linewidths=.5, linecolor="black", cmap="Greys", cbar=True)
    sns.heatmap(x, ax=ax3, vmin=0, vmax=1, linewidths=.5, linecolor="black", cmap="Greys", cbar=True)

    plt.tight_layout()
    plt.savefig(image_dir+"/result_"+experiment_name+"_{}.png".format(epoch), dpi=250)

    #fig = plt.figure(figsize=(15,10))
    fig = plt.figure()
    ax1_2 = fig.add_subplot(321)
    ax2_2 = fig.add_subplot(325)
    ax3_2 = fig.add_subplot(322)
    ax4_2 = fig.add_subplot(324)
    ax5_2 = fig.add_subplot(326)
    ax6_2 = fig.add_subplot(323)
    ax1_2.set_title("Read Weigths")
    ax2_2.set_title("Write Weights")
    ax3_2.set_title("Forward Mode")
    ax4_2.set_title("Content Mode")
    ax5_2.set_title("Backward Mode")
    ax6_2.set_title("Read Modes")
    ax6_2.set_yticklabels(["back", "forw", "cont"])

    sns.heatmap(v['read_weights'].T, ax=ax1_2, linewidths=.01)
    sns.heatmap(v['write_weights'].T, ax=ax2_2, linewidths=.01)
    sns.heatmap(v['forward_mode'].T, ax=ax3_2, linewidths=.01)
    sns.heatmap(v['content_mode'].T, ax=ax4_2, linewidths=.01)
    sns.heatmap(v['backward_mode'].T, ax=ax5_2, linewidths=.01)
    sns.heatmap(v['read_modes'].T, ax=ax6_2, linewidths=.01)

    plt.tight_layout()
    plt.savefig(image_dir+"/weights_"+experiment_name+"_{}.png".format(epoch), dpi=250)
