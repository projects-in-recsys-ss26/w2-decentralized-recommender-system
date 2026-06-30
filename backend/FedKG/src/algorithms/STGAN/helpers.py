import torch
from torch.autograd import Variable
from math import ceil



def prepare_generator_batch(samples, start_letter=0, device='cuda'):

    batch_size, seq_len = samples.size()

    inp = torch.zeros(batch_size, seq_len, device=device).long()  # Initialize inp directly on the correct device
    target = samples.to(device)  # Ensure target is on the correct device
    inp[:, 0] = start_letter
    inp[:, 1:] = target[:, :seq_len-1].long()  # Ensure the slice is also long type

    # The Variables are deprecated and no longer needed here
    # The to(device) is already handled above when initializing and copying tensors

    return inp, target


import torch

def prepare_discriminator_data(pos_samples, neg_samples, device='cuda'):
    """
    Build the discriminator's input and target tensors from positive samples
    (real/target data) and negative samples (generator output).
    """

    # Concatenate positive and negative samples, cast to long, move to device
    inp = torch.cat((pos_samples, neg_samples), 0).long().to(device)
    # Target tensor of all ones, length = #positive + #negative, on the device
    target = torch.ones(pos_samples.size(0) + neg_samples.size(0)).to(device)
    # Set the negative-sample portion of the target to 0
    target[pos_samples.size(0):] = 0

    # Shuffle the data
    perm = torch.randperm(target.size(0)).to(device)
    target = target[perm]
    inp = inp[perm]

    return inp, target


def batchwise_sample(gen, num_samples, batch_size):
    """
    Sample num_samples samples batch_size samples at a time from gen.
    Does not require gpu since gen.sample() takes care of that.
    """

    samples = []
    for i in range(int(ceil(num_samples/float(batch_size)))):
        samples.append(gen.sample(batch_size))

    return torch.cat(samples, 0)[:num_samples]


def batchwise_oracle_nll(gen, oracle, num_samples, batch_size, max_seq_len, start_letter=0, device='cuda'):
    s = batchwise_sample(gen, num_samples, batch_size)
    oracle_nll = 0
    for i in range(0, num_samples, batch_size):
        inp, target = prepare_generator_batch(s[i:i+batch_size], start_letter, device)
        oracle_loss = oracle.batchNLLLoss(inp, target) / max_seq_len
        oracle_nll += oracle_loss.data.item()

    return oracle_nll/(num_samples/batch_size)
