import sys

import numpy as np

import torch
import torch as T
import torch.nn.functional as F

def llprint(message):
    sys.stdout.write(message)
    sys.stdout.flush()

def generate_data(batch_size, length, size, steps=0, cuda=-1, non_uniform=False):

    # Generate the binary sequences of length equal to size.
    # We leave 2 bits empty for the priority and the delimiter.
    # Moreover, we add an empty vector which will be used as signal
    # for printing the result
    seq = np.random.binomial(1, 0.5, (batch_size, length, size - 2))
    seq = torch.from_numpy(seq)

    # Add priority number (just a single one drawn from the uniform distribution
    # or from the beta distribution)
    if not non_uniform:
        priority = np.random.uniform(-1, 1, (batch_size, length, 1))
    else:
        priority = np.random.beta(1,3, (batch_size, length, 1))
    priority = torch.from_numpy(priority)

    # Generate the first tensor
    inp = torch.zeros(batch_size, (length + 1), size)
    inp[:, :(length), :(size - 2)] = seq
    inp[:, :(length), (size - 2):] = priority
    inp[:, :(length), (size - 1):] = torch.zeros(batch_size, length, 1)

    # If the length is just 1, then add the delimiter
    if (steps==0):
        inp[:, length, size - 1] = 1

    # For each step, we add an empty space
    for s in range(0,steps):
        inp_tmp = torch.zeros(batch_size, (1), size)

        # If this is the last repetition then we set the bit to 1.
        if (s == steps-1):
            inp_tmp[:, 0, size-1] = 1

        # Concatenate the tensor to the previous one
        inp = torch.cat((inp, inp_tmp), 1)

    # We then add an empty section in which we need to store the result
    # only if the steps are greater than 1
    if steps > 0:
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
        temp.sort(key=lambda x: x[size-2], reverse=True)  # Sort elements descending order
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

    return inp.float(), outp.float()[:, :-1, :]

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
