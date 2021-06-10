# Copyright (c) Microsoft. All rights reserved.
import random
import torch
import numpy
from torch.autograd import Variable
import subprocess
import matplotlib.pyplot as plt

class AverageMeter(object):
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def set_environment(seed, set_cuda=False):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available() and set_cuda:
        torch.cuda.manual_seed_all(seed)

def patch_var(v, cuda=True):
    if cuda:
        v = Variable(v.cuda(non_blocking=True))
    else:
        v = Variable(v)
    return v

def get_gpu_memory_map():
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map

def get_pip_env():
    result = subprocess.call(["pip", "freeze"])
    return result

def plot(fig,ax,data, fname, label = None, xlabel = 'epochs', ylabel = 'dev metrics'):

    #fig = plt.figure(figsize=(6, 3))
    #ax = fig.add_subplot(111)
    x = numpy.arange(len(data))
    ax.plot(x, data, label=label)
    ax.legend(loc='best', fontsize=11)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.tick_params(direction='in')
    fig.tight_layout()
    fig.savefig('images/'+ fname+'.pdf')
