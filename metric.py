import torch
import numpy as np

def nrmse(y_predicted, y):
    var = torch.std(y)**2
    error = (y - y_predicted)**2
    return float(torch.sqrt(torch.mean(error) / var))

def kernel_quality(i, esn, ks):
    # «Connectivity, Dynamics and Memory in Reservoir Computing with Binary and
    # Analog Neurons».
    inputs = torch.rand(i*ks)*2 - 1

    split_overflow = len(inputs) % ks
    if split_overflow != 0:
        inputs = inputs[:-split_overflow]
    us = np.split(inputs.numpy(), ks)
    us = torch.FloatTensor(np.array(us))  # Convert list to numpy array first, then to tensor

    M = [0]*ks
    for i, u in enumerate(us):
        esn(u, kq=True)
        M[i] = np.array(esn.X[-1])

    kq = np.linalg.matrix_rank(M)
    return kq

def generalization(i, esn, ks):
    # «Connectivity, Dynamics and Memory in Reservoir Computing with Binary and
    # Analog Neurons». Footnote 5.
    inputs = np.random.rand(i*ks)*2 - 1

    split_overflow = len(inputs) % ks
    if split_overflow != 0:
        inputs = inputs[:-split_overflow]
    us = np.split(inputs, ks)

    # Generalization part: set the last fifth of the input stream to always be
    # the same.
    gen_length = i // 5
    gen_seq = us[0][-gen_length:]
    for u in us:
        np.put(u, np.arange(-gen_length, 0), gen_seq)
    us = torch.FloatTensor(np.array(us))  # Convert list to numpy array first, then to tensor

    M = [0]*ks
    for i, u in enumerate(us):
        esn(u, kq=True)
        M[i] = np.array(esn.X[-1])

    kq = np.linalg.matrix_rank(M)
    return kq

def memory_capacity(esn):
    # Generated according to «Computational analysis of memory capacity in echo
    # state networks», discarding 100 for transients (washout), using 1100 for
    # training and 1000 for testing the memory capacity.
    torch.manual_seed(0)
    inputs = torch.FloatTensor(2200).uniform_(-1, 1)
    washout = inputs[:100]
    u_train = inputs[100:1200]
    u_test = inputs[1200:]
    return esn.memory_capacity(washout, u_train, u_test, plot=False)