import torch
import numpy as np
from torch.autograd import Variable as var

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

    # As final step, we add to the final input the sequence
    inp_tmp = torch.zeros(batch_size, (length + 1), size)
    inp_tmp[:, :(length), :(size - 2)] = seq
    inp_tmp[:, :(length), (size - 2):] = priority
    inp_tmp[:, :(length), (size - 1):] = torch.zeros(batch_size, length, 1)
    inp = torch.cat((inp, inp_tmp), 1)

    outp = inp.numpy()

    # Strip all the binary vectors into a list
    # and sort the list by looking at the last column
    # (which will contain the priority)
    temp = []
    for i in range(length):
        temp.append(outp[0][i])
    temp.sort(key=lambda x: x[size-2], reverse=True)  # Sort elements descending order

    # FIXME
    # Ugly hack to present the tensor structure as the one
    # required by the framework
    layer = []
    for i in range(len(temp)):
        layer.append(np.array(temp[i]))
    output_final = []
    output_final.append(layer)

    # Convert everything to numpy and to a tensor
    outp = torch.from_numpy(np.array(output_final))

    # Add an empy line at the end to simulate the delimiter
    outp = torch.cat((outp, torch.zeros(batch_size, 1, size)),1)

    if cuda != -1:
        inp = inp.cuda()
        outp = outp.cuda()

    return var(inp.float()), var(outp.float())

if __name__ == "__main__":

    input, output = generate_data(1, 2, 10, 1)
    print(input)
    print(input.shape)
    print(output)
