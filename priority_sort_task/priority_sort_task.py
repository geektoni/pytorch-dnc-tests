#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import argparse
from visdom import Visdom

from utils import *

import json

from dnc.dnc import DNC
from dnc.sdnc import SDNC
from dnc.sam import SAM
from dnc.util import *

import torch.autograd as autograd
import torch.optim as optim

from tensorboard_logger import configure, log_value
import seaborn as sns
from tqdm import tqdm

if __name__ == "__main__":

    # Set the arg parser
    parser = argparse.ArgumentParser(description='PyTorch DNC Priority Sort Task')
    parser.add_argument('-input_size', type=int, default=8, help='dimension of input feature')
    parser.add_argument('-rnn_type', type=str, default='lstm', help='type of recurrent cells to use for the controller')
    parser.add_argument('-nhid', type=int, default=100, help='number of hidden units of the inner nn')
    parser.add_argument('-dropout', type=float, default=0, help='controller dropout')
    parser.add_argument('-memory_type', type=str, default='dnc', help='dense or sparse memory: dnc | sdnc | sam')
    parser.add_argument('-tb_dir', type=str, default='./tensorboard', help='tensorboard log directory')

    parser.add_argument('-steps', type=int, default=1, help="Number of steps we give to the DNC")
    parser.add_argument('-checkpoint_dir', type=str, default='checkpoints', help="Name of the directory where we will save the checkpoints")
    parser.add_argument('-independent_linears', action='store_true', help="Use independent linears for the memory management")

    parser.add_argument('-nlayer', type=int, default=1, help='number of layers')
    parser.add_argument('-nhlayer', type=int, default=2, help='number of hidden layers')
    parser.add_argument('-lr', type=float, default=3e-5, help='initial learning rate')
    parser.add_argument('-optim', type=str, default='rmsprop', help='learning rule, supports adam|rmsprop')
    parser.add_argument('-clip', type=float, default=10, help='gradient clipping')

    parser.add_argument('-batch_size', type=int, default=100, metavar='N', help='batch size')
    parser.add_argument('-mem_size', type=int, default=20, help='memory dimension')
    parser.add_argument('-mem_slot', type=int, default=16, help='number of memory slots')
    parser.add_argument('-read_heads', type=int, default=5, help='number of read heads')
    parser.add_argument('-sparse_reads', type=int, default=10, help='number of sparse reads per read head')
    parser.add_argument('-temporal_reads', type=int, default=2, help='number of temporal reads')

    parser.add_argument('-sequence_max_length', type=int, default=20, metavar='N', help='sequence_max_length')
    parser.add_argument('-curriculum_increment', type=int, default=0, metavar='N',
                        help='sequence_max_length incrementor per 1K iterations')
    parser.add_argument('-curriculum_freq', type=int, default=1000, metavar='N',
                        help='sequence_max_length incrementor per 1K iterations')
    parser.add_argument('-cuda', type=int, default=-1, help='Cuda GPU ID, -1 for CPU')

    parser.add_argument('-iterations', type=int, default=1000000, metavar='N', help='total number of iteration')
    parser.add_argument('-summarize_freq', type=int, default=100, metavar='N', help='summarize frequency')
    parser.add_argument('-check_freq', type=int, default=100000, metavar='N', help='check point frequency')
    parser.add_argument('-visdom', action='store_true', help='plot memory content on visdom per -summarize_freq steps')
    parser.add_argument('-non_uniform_priority', action="store_true", help='Draw the priority value from the beta distribution')

    args = parser.parse_args()
    print(args)

    # Generate the name of this experiment
    experiment_name = "priority_sort_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(
        args.input_size,
        args.rnn_type,
        args.nhid,
        args.memory_type,
        args.steps,
        args.batch_size,
        args.mem_size,
        args.mem_slot,
        args.sequence_max_length,
        args.iterations,
        args.non_uniform_priority
    )

    # Add directory were to save the tensorboard logs
    configure(args.tb_dir+"/"+experiment_name)

    # Set the random seed used
    np.random.seed(42)
    T.manual_seed(42)

    # Enable visdom if we need it
    if args.visdom:
        viz = Visdom()

    # Setup the directory were we will save the checkpoints
    dirname = os.path.dirname(".")
    ckpts_dir = os.path.join(dirname, args.checkpoint_dir)+"/"+experiment_name
    if not os.path.isdir(ckpts_dir):
        os.makedirs(ckpts_dir)
        os.makedirs(ckpts_dir+"/images")

    # Setup basic informations
    batch_size = args.batch_size
    sequence_max_length = args.sequence_max_length
    iterations = args.iterations
    summarize_freq = args.summarize_freq
    check_freq = args.check_freq

    # input_size = output_size = args.input_size
    mem_slot = args.mem_slot
    mem_size = args.mem_size
    read_heads = args.read_heads

    independent_linears=False
    if args.independent_linears:
        independent_linears=args.independent_linears

    if args.memory_type == 'dnc':
        rnn = DNC(
        input_size=args.input_size+3,
        hidden_size=args.nhid,
        rnn_type=args.rnn_type,
        num_layers=args.nlayer,
        num_hidden_layers=args.nhlayer,
        dropout=args.dropout,
        nr_cells=mem_slot,
        cell_size=mem_size,
        read_heads=read_heads,
        gpu_id=args.cuda,
        debug=True,
        batch_first=True,
        independent_linears=independent_linears
    )
    elif args.memory_type == 'sdnc':
        rnn = SDNC(
        input_size=args.input_size+3,
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
        independent_linears=independent_linears
    )
    elif args.memory_type == 'sam':
        rnn = SAM(
        input_size=args.input_size+3,
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
        independent_linears=independent_linears
    )
    else:
        raise Exception('Not recognized type of memory')

    # Print the structure of the rnn
    print(rnn)

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

    # Create the loss object
    bce_loss = nn.BCELoss(reduction='mean')
    sigm = nn.Sigmoid()

    (chx, mhx, rv) = (None, None, None)
    for epoch in tqdm(range(iterations)):
        optimizer.zero_grad()

        #random_length = np.random.randint(1, sequence_max_length + 1)
        random_length=sequence_max_length

        # Use the input size given by the user and increment it by 2 in order to
        # add space for the priority and the delimiter
        input_data, target_output = generate_data(batch_size, random_length, args.input_size+3, cuda=args.cuda, steps=args.steps, non_uniform=args.non_uniform_priority)

        with autograd.detect_anomaly():

            if rnn.debug:
                output, (chx, mhx, rv), v = rnn(input_data, (None, mhx, None), reset_experience=True, pass_through_memory=True)
            else:
                output, (chx, mhx, rv) = rnn(input_data, (None, mhx, None), reset_experience=True, pass_through_memory=True)


            # We compute the loss by taking into account only the vectors and not
            # the delimiter bit or the priority. This has to be done in order to
            # have a negative loss.
            if args.steps == 0:
                loss = bce_loss(sigm(output[:, :-1, :-3]), target_output[:,:,:-3])
            else:
                loss = bce_loss(sigm(output[:, ((random_length+1)+args.steps+1):, :-3]), target_output[:,:,:-3])

            loss.backward()

            T.nn.utils.clip_grad_norm_(rnn.parameters(), args.clip)
            optimizer.step()
            loss_value = loss.item()

            summarize = (epoch % summarize_freq == 0) and (epoch != 0)
            take_checkpoint = (epoch != 0) and (epoch % check_freq == 0)

            # Increment size just if we have set a curriculum frequency
            increment_curriculum = False
            if args.curriculum_freq != 0:
                increment_curriculum = (epoch != 0) and (epoch % args.curriculum_freq == 0)

            # detach memory from graph
            mhx = { k : v.detach() for k, v in mhx.items() }

            # Save loss value
            save_losses.append(loss_value)
            last_losses.append(loss_value)

            # Save cost value
            if args.steps==0:
                current_cost = compute_cost(sigm(output[:, :-1, :-3]), target_output[:,:,:-3], batch_size=batch_size).item()
            else:
                current_cost = compute_cost(sigm(output[:, ((random_length+1)+args.steps+1):, :-3]), target_output[:, :, :-3], batch_size=batch_size).item()
            costs.append(current_cost)
            last_costs.append(current_cost)

            if summarize:
                llprint("\n\t[*] Iteration {ep}/{tot}".format(ep=epoch, tot=iterations))
                loss = np.mean(last_losses)
                cost = np.mean(last_costs)
                last_losses = []
                last_costs = []
                llprint("\n\t[*] Avg. Logistic Loss: %.4f" % (loss))
                llprint("\n\t[*] Avg. Cost: %4f\n"% (cost))
                if np.isnan(loss):
                    raise Exception('We computed a NaN Loss')

            if summarize and rnn.debug:
                loss = np.mean(last_losses)
                last_losses = []
                last_costs = []

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

            # Increment the size of the input
            if increment_curriculum:
                sequence_max_length = sequence_max_length + args.curriculum_increment
                llprint("\n[*] Increasing max length to " + str(sequence_max_length)+"\n")

            # Take a checkpoint, save the model and the data
            if take_checkpoint:

                check_ptr = os.path.join(ckpts_dir, "model_"+experiment_name+".pt.{}".format(epoch))
                llprint("\n[*] Saving Checkpoint to {}\n".format(check_ptr))
                cur_weights = rnn.state_dict()
                T.save(cur_weights, check_ptr)

                # Generate images
                generate_result_images(
                    sigm(output).detach().numpy(),target_output.detach().numpy(),
                    v['read_weights'],v['write_weights'],
                    ckpts_dir+"/images",
                    experiment_name,
                    epoch,
                    args,
                    check_ptr)

                # Save data
                performance_data_path = os.path.join(ckpts_dir, "results_"+experiment_name+"_{}.json".format(epoch))
                content = {
                  "loss": save_losses,
                  "cost": costs
                }
                f = open(performance_data_path, 'w+')
                f.write(json.dumps(content))
                f.close()

                llprint("Check point done!\n")
