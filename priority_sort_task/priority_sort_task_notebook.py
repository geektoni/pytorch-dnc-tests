# -*- coding: utf-8 -*-
#import warnings
#warnings.filterwarnings('ignore')

import numpy as np
import getopt
import sys
import os
import math
import time
import argparse
from visdom import Visdom

import json

sys.path.insert(0, os.path.join('..', '..'))

import torch as T
from torch.autograd import Variable as var
import torch.nn.functional as F
import torch.optim as optim

from torch.nn.utils import clip_grad_norm_

from dnc.dnc import DNC
from dnc.sdnc import SDNC
from dnc.sam import SAM
from dnc.util import *

np.random.seed(42)
T.manual_seed(42)

def llprint(message):
    sys.stdout.write(message)
    sys.stdout.flush()

def UNgenerate_data(batch_size, length, size, cuda=-1):

    print(batch_size, length, size)

    input_data = np.zeros((batch_size, 2 * length + 1, size), dtype=np.float32)
    target_output = np.zeros((batch_size, 2 * length + 1, size), dtype=np.float32)

    sequence = np.random.binomial(1, 0.5, (batch_size, length, size - 1))

    input_data[:, :length, :size - 1] = sequence
    input_data[:, length, -1] = 1  # the end symbol
    target_output[:, length + 1:, :size - 1] = sequence

    input_data = T.from_numpy(input_data)
    target_output = T.from_numpy(target_output)
    if cuda != -1:
        input_data = input_data.cuda()
        target_output = target_output.cuda()
    return var(input_data), var(target_output)


def generate_data(batch_size, length, size, steps=1, cuda=-1):

    # Generate the binary sequences of length equal to size.
    # We leave 2 bits empty for the priority and the delimiter.
    # Moreover, we add an empty vector which will be used as signal
    # for printing the result
    seq = np.random.binomial(1, 0.5, (batch_size, length, size - 2))
    seq = torch.from_numpy(seq)

    # Add priority number (just a single one drawn from the uniform distribution)
    priority = np.random.uniform(-1, 1, (batch_size, length, 1))
    priority = torch.from_numpy(priority)

    # Generate the first tensor
    inp = torch.zeros(batch_size, (length + 1), size)
    inp[:, :(length), :(size - 2)] = seq
    inp[:, :(length), (size - 2):] = priority
    inp[:, :(length), (size - 1):] = torch.zeros(batch_size, length, 1)

    # If the length is just 1, then add the delimiter
    if (steps==1):
        inp[:, length, size - 1] = 1

    # For each step, we add a copy of the sequence
    for s in range(2,steps+1):
        inp_tmp = torch.zeros(batch_size, (length + 1), size)
        inp_tmp[:, :(length), :(size-2)] = seq
        inp_tmp[:, :(length), (size-2):] = priority
        inp_tmp[:, :(length), (size-1):] = torch.zeros(batch_size, length, 1)

        # If this is the last repetition then we set the bit to 1
        if (s == steps):
            inp_tmp[:, length, size-1] = 1

        # Concatenate the tensor to the previous one
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

    return var(inp.float()), var(outp.float())

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

def compute_cost(output, target_out):

  # Binarize the result
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

from argparse import Namespace

args = Namespace(input_size=10, rnn_type="lstm", nhid=100, dropout=0, memory_type="dnc", nlayer=1, nhlayer=2,
                 lr=3e-5, optim="rmsprop", clip=10, batch_size=1, mem_size=20, mem_slot=128, read_heads=5,
                 sparse_reads=10, temporal_reads=2, sequence_max_length=20, curriculum_increment=0, curriculum_freq=1000,
                 cuda=-1, iterations=1000000, summarize_freq=100, check_freq=100000, visdom=True)
if args.visdom:
    viz = Visdom()

dirname = os.path.dirname(".")
ckpts_dir = os.path.join(dirname, 'checkpoints')
if not os.path.isdir(ckpts_dir):
    os.mkdir(ckpts_dir)

batch_size = args.batch_size
sequence_max_length = args.sequence_max_length
iterations = args.iterations
summarize_freq = args.summarize_freq
check_freq = args.check_freq

# input_size = output_size = args.input_size
mem_slot = args.mem_slot
mem_size = args.mem_size
read_heads = args.read_heads

if args.memory_type == 'dnc':
    rnn = DNC(
    input_size=args.input_size,
    hidden_size=args.nhid,
    rnn_type=args.rnn_type,
    num_layers=args.nlayer,
    num_hidden_layers=args.nhlayer,
    dropout=args.dropout,
    nr_cells=mem_slot,
    cell_size=mem_size,
    read_heads=read_heads,
    gpu_id=args.cuda,
    debug=args.visdom,
    batch_first=True,
    independent_linears=True
)
elif args.memory_type == 'sdnc':
    rnn = SDNC(
    input_size=args.input_size,
    hidden_size=args.nhid,
    rnn_type=args.rnn_type,
    num_layers=args.nlayer,
    num_hidden_layers=args.nhlayer,
    dropout=args.dropout,
    nr_cells=mem_slot,
    cell_size=mem_size,
    sparse_reads=args.sparse_reads,
    temporal_reads=args.temporal_reads,
    read_heads=args.read_heads,
    gpu_id=args.cuda,
    debug=args.visdom,
    batch_first=True,
    independent_linears=False
)
elif args.memory_type == 'sam':
    rnn = SAM(
    input_size=args.input_size,
    hidden_size=args.nhid,
    rnn_type=args.rnn_type,
    num_layers=args.nlayer,
    num_hidden_layers=args.nhlayer,
    dropout=args.dropout,
    nr_cells=mem_slot,
    cell_size=mem_size,
    sparse_reads=args.sparse_reads,
    read_heads=args.read_heads,
    gpu_id=args.cuda,
    debug=args.visdom,
    batch_first=True,
    independent_linears=False
)
else:
    raise Exception('Not recognized type of memory')

print(rnn)
# register_nan_checks(rnn)

if args.cuda != -1:
    rnn = rnn.cuda(args.cuda)

if args.optim == 'adam':
    optimizer = optim.Adam(rnn.parameters(), lr=args.lr, eps=1e-9, betas=[0.9, 0.98]) # 0.0001
elif args.optim == 'adamax':
    optimizer = optim.Adamax(rnn.parameters(), lr=args.lr, eps=1e-9, betas=[0.9, 0.98]) # 0.0001
elif args.optim == 'rmsprop':
    optimizer = optim.RMSprop(rnn.parameters(), lr=args.lr, momentum=0.9, eps=1e-10) # 0.0001
elif args.optim == 'sgd':
    optimizer = optim.SGD(rnn.parameters(), lr=args.lr) # 0.01
elif args.optim == 'adagrad':
    optimizer = optim.Adagrad(rnn.parameters(), lr=args.lr)
elif args.optim == 'adadelta':
    optimizer = optim.Adadelta(rnn.parameters(), lr=args.lr)


# List for keeping useful data
costs = []
last_costs = []
last_losses = []
save_losses= []
seq_lengths = []

(chx, mhx, rv) = (None, None, None)
for epoch in range(iterations + 1):
    llprint("\rIteration {ep}/{tot}".format(ep=epoch, tot=iterations))
    optimizer.zero_grad()

    #random_length = np.random.randint(1, sequence_max_length + 1)
    random_length=sequence_max_length

    input_data, target_output = generate_data(batch_size, random_length, args.input_size, cuda=args.cuda)

    if rnn.debug:
        output, (chx, mhx, rv), v = rnn(input_data, (None, mhx, None), reset_experience=True, pass_through_memory=True)
    else:
        output, (chx, mhx, rv) = rnn(input_data, (None, mhx, None), reset_experience=True, pass_through_memory=True)

    loss = criterion((output), target_output)

    loss.backward()

    T.nn.utils.clip_grad_norm_(rnn.parameters(), args.clip)
    optimizer.step()
    loss_value = loss.item()

    summarize = (epoch % summarize_freq == 0)
    take_checkpoint = (epoch != 0) and (epoch % check_freq == 0)
    increment_curriculum = (epoch != 0) and (epoch % args.curriculum_freq == 0)

    # detach memory from graph
    mhx = { k : (v.detach() if isinstance(v, var) else v) for k, v in mhx.items() }

    # Save loss value
    save_losses.append(loss_value)
    last_losses.append(loss_value)

    # Save cost value
    current_cost = compute_cost(output, target_output).item()
    costs.append(current_cost)
    last_costs.append(current_cost)

    # Save sequence length
    seq_lengths.append(args.input_size)

    if summarize:
        loss = np.mean(last_losses)
        cost = np.mean(last_costs)
        last_losses = []
        last_costs = []
      # print(input_data)
      # print("1111111111111111111111111111111111111111111111")
      # print(target_output)
      # print('2222222222222222222222222222222222222222222222')
      # print(F.relu6(output))
        llprint("\n\tAvg. Logistic Loss: %.4f" % (loss))
        llprint("\n\tAvg. Cost: %4f\n"% (cost))
        if np.isnan(loss):
            raise Exception('nan Loss')

    if summarize and rnn.debug:
        loss = np.mean(last_losses)
        last_losses = []
        last_costs = []
      # print(input_data)
      # print("1111111111111111111111111111111111111111111111")
      # print(target_output)
      # print('2222222222222222222222222222222222222222222222')
      # print(F.relu6(output))
        #last_save_losses = []

    if args.visdom:
        if args.memory_type == 'dnc':
            viz.heatmap(
                v['memory'],
                opts=dict(
                    xtickstep=10,
                    ytickstep=2,
                    title='Memory, t: ' + str(epoch) + ', loss: ' + str(loss),
                    ylabel='layer * time',
                    xlabel='mem_slot * mem_size'
                )
            )

        if args.memory_type == 'dnc':
            viz.heatmap(
                v['link_matrix'][-1].reshape(args.mem_slot, args.mem_slot),
                opts=dict(
                    xtickstep=10,
                    ytickstep=2,
                    title='Link Matrix, t: ' + str(epoch) + ', loss: ' + str(loss),
                    ylabel='mem_slot',
                    xlabel='mem_slot'
                )
            )
        elif args.memory_type == 'sdnc':
            viz.heatmap(
                v['link_matrix'][-1].reshape(args.mem_slot, -1),
                opts=dict(
                    xtickstep=10,
                    ytickstep=2,
                    title='Link Matrix, t: ' + str(epoch) + ', loss: ' + str(loss),
                    ylabel='mem_slot',
                    xlabel='mem_slot'
                )
            )

            viz.heatmap(
                v['rev_link_matrix'][-1].reshape(args.mem_slot, -1),
                opts=dict(
                    xtickstep=10,
                    ytickstep=2,
                    title='Reverse Link Matrix, t: ' + str(epoch) + ', loss: ' + str(loss),
                    ylabel='mem_slot',
                    xlabel='mem_slot'
                )
            )

        elif args.memory_type == 'sdnc' or args.memory_type == 'dnc':
            viz.heatmap(
                v['precedence'],
                opts=dict(
                    xtickstep=10,
                    ytickstep=2,
                    title='Precedence, t: ' + str(epoch) + ', loss: ' + str(loss),
                    ylabel='layer * time',
                    xlabel='mem_slot'
                )
            )

        if args.memory_type == 'sdnc':
            viz.heatmap(
                v['read_positions'],
                opts=dict(
                    xtickstep=10,
                    ytickstep=2,
                    title='Read Positions, t: ' + str(epoch) + ', loss: ' + str(loss),
                    ylabel='layer * time',
                    xlabel='mem_slot'
                )
            )

            viz.heatmap(
              v['read_weights'],
              opts=dict(
                  xtickstep=10,
                  ytickstep=2,
                  title='Read Weights, t: ' + str(epoch) + ', loss: ' + str(loss),
                  ylabel='layer * time',
                  xlabel='nr_read_heads * mem_slot'
              )
          )

            viz.heatmap(
              v['write_weights'],
              opts=dict(
                  xtickstep=10,
                  ytickstep=2,
                  title='Write Weights, t: ' + str(epoch) + ', loss: ' + str(loss),
                  ylabel='layer * time',
                  xlabel='mem_slot'
              )
          )

            viz.heatmap(
              v['usage_vector'] if args.memory_type == 'dnc' else v['usage'],
              opts=dict(
                  xtickstep=10,
                  ytickstep=2,
                  title='Usage Vector, t: ' + str(epoch) + ', loss: ' + str(loss),
                  ylabel='layer * time',
                  xlabel='mem_slot'
              )
          )

    if increment_curriculum:
        sequence_max_length = sequence_max_length + args.curriculum_increment
        print("Increasing max length to " + str(sequence_max_length))

    if take_checkpoint:
        check_ptr = os.path.join(ckpts_dir, 'step_{}.pth'.format(epoch))
        llprint("\nSaving Checkpoint to {}\n".format(check_ptr))
        cur_weights = rnn.state_dict()
        T.save(cur_weights, check_ptr)

        # Save data
        performance_data_path = os.path.join(ckpts_dir, 'results_{}.csv'.format(epoch))
        content = {
          "loss": save_losses,
          "cost": costs,
          "seq_lengths": seq_lengths
        }
        f = open(performance_data_path, 'w+')
        f.write(json.dumps(content))
        f.close()

        llprint("Done!\n")



for i in range(int((iterations + 1) / 10)):
    llprint("\nIteration %d/%d" % (i, iterations))
    # We test now the learned generalization using sequence_max_length examples
    random_length = np.random.randint(2, sequence_max_length * 10 + 1)
    input_data, target_output, loss_weights = generate_data(random_length, input_size)

    if rnn.debug:
        output, (chx, mhx, rv), v = rnn(input_data, (None, mhx, None), reset_experience=True, pass_through_memory=True)
    else:
        output, (chx, mhx, rv) = rnn(input_data, (None, mhx, None), reset_experience=True, pass_through_memory=True)

    output = output[:, -1, :].sum().data.cpu().numpy()[0]
    target_output = target_output.sum().data.cpu().numpy()

try:
    print("\nReal value: ", ' = ' + str(int(target_output[0])))
    print("Predicted:  ", ' = ' + str(int(output // 1)) + " [" + str(output) + "]")
except Exception as e:
    pass
