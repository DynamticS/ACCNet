import os
import sys
import time
import h5py
import numpy as np
import pprint
import random
import torch
import torch.fft as fft
import layers

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import torch.nn as nn

from scipy.sparse import coo_matrix
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader

def set_gpu(x):
    torch.set_num_threads(1)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(x)
    print('using gpu:', x)


def seed_all(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)


def ensure_path(path):
    if os.path.exists(path):
        pass
    else:
        os.makedirs(path)


class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    return (pred == label).type(torch.cuda.FloatTensor).mean().item()


class Timer():

    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)

_utils_pp = pprint.PrettyPrinter()
def pprint(x):
    _utils_pp.pprint(x)


def get_model(args):


    if args.model == 'ACCNet':

        if args.data_using =='FACED':
            in_channels = 500
            hidden_channels = 128

        model = layers.ACCNet(args, in_channels=in_channels, hidden_channels=hidden_channels, out_channels=2)


    return model


def get_dataloader(args, data, label, batch_size):
    # load the data
    # dataset = [Data(x=data[i], edge_index=get_edge_index(args), y=label[i]) for i in range(data.shape[0])]
    dataset = [Data(x=data[i], edge_index=get_edge_index(args), y=label[i]) for i in range(data.shape[0])]
    print(f"---Initialize the {args.data_using} data into graph-based data---")
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    return loader
def get_edge_index(args):
    if args.dense == 'full' and args.data_using == 'DEAP':
        nodes_num = 32
        edge_data = torch.ones([nodes_num, nodes_num])  # initialize edge index
        edge_index = coo_matrix(edge_data)
        edge_index = np.vstack((edge_index.row, edge_index.col))
        # edge_index = torch.from_numpy(edge_index).to(torch.int64).to(device)
        edge_index = torch.from_numpy(edge_index).to(torch.int64)

    if args.dense == 'full' and args.data_using == 'FACED':
        nodes_num = 32
        edge_data = torch.ones([nodes_num, nodes_num])  # initialize edge index
        edge_index = coo_matrix(edge_data)
        edge_index = np.vstack((edge_index.row, edge_index.col))
        # edge_index = torch.from_numpy(edge_index).to(torch.int64).to(device)
        edge_index = torch.from_numpy(edge_index).to(torch.int64)

    return edge_index

def get_edge_attr(type, signal_patch, batch):
    CUDA = torch.cuda.is_available()
    if type == 'COS':

        signal_patch = signal_patch.reshape(batch, signal_patch.shape[0] // batch,
                                            signal_patch.shape[-1])
        norms = torch.norm(signal_patch, dim=2, keepdim=True)
        signal_patch_normalized = signal_patch / (norms + 1e-8)
        sim_matrices = torch.bmm(signal_patch, signal_patch_normalized.transpose(1,2))
        edge_attr = sim_matrices.view(-1)
        edge_attr_2 = (edge_attr - edge_attr.min()) / (edge_attr.max() - edge_attr.min())
        edge_attr = edge_attr_2

    if type == 'PLI':

        signal_patch = signal_patch.reshape(batch, signal_patch.shape[0] // batch,
                                            signal_patch.shape[-1])

        phase_diff_matrices = calculate_phase_difference_matrix(signal_patch)

        normalized_matrices = sigmoid_normalize_phase_diff_matrix(phase_diff_matrices)
        edge_attr = normalized_matrices.view(-1)   



    return  edge_attr




def get_metrics(y_pred, y_true, classes=None):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    if classes is not None:
        cm = confusion_matrix(y_true, y_pred, labels=classes)
    else:
        cm = confusion_matrix(y_true, y_pred)
    return acc, f1, cm


def get_trainable_parameter_num(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params


def L1Loss(model, Lambda):
    w = torch.cat([x.view(-1) for x in model.parameters()])
    err = Lambda * torch.sum(torch.abs(w))
    return err


def L2Loss(model, Lambda):
    w = torch.cat([x.view(-1) for x in model.parameters()])
    err = Lambda * torch.sum(w.pow(2))
    return err


class LabelSmoothing(nn.Module):
    """NLL loss with label smoothing.
       refer to: https://github.com/NVIDIA/DeepLearningExamples/blob/8d8b21a933fff3defb692e0527fca15532da5dc6/PyTorch/Classification/ConvNets/image_classification/smoothing.py#L18
    """
    def __init__(self, smoothing=0.0):
        """Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()




def calculate_phase_difference_matrix(signals):

    analytic_signals = hilbert_torch(signals)
    phases = torch.angle(analytic_signals)  
    phase_diff = phases.unsqueeze(2) - phases.unsqueeze(1)  
    phase_diff = torch.atan2(torch.sin(phase_diff), torch.cos(phase_diff))
    phase_diff_matrix = phase_diff.mean(dim=-1) 
    
    return phase_diff_matrix

def sigmoid_normalize_phase_diff_matrix(phase_diff_matrix):
  
    return torch.sigmoid(phase_diff_matrix)

def hilbert_torch(x):

    N = x.shape[-1]
    Xf = fft.fft(x, dim=-1)
    h = torch.zeros(N, dtype=torch.complex64, device=x.device)
    if N % 2 == 0:
        h[0] = h[N // 2] = 1
        h[1:N // 2] = 2
    else:
        h[0] = 1
        h[1:(N + 1) // 2] = 2
    return fft.ifft(Xf * h, dim=-1)