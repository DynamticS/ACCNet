import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import coo_matrix
from torch_geometric.nn import GCNConv, global_mean_pool,GATv2Conv
from torch_geometric.data import Data, DataLoader
from sklearn.model_selection import train_test_split
criterion = torch.nn.CrossEntropyLoss()
import utils as ut
import numpy as np
import argparse
import torch.fft as fft

old_repr = torch.Tensor.__repr__
def tensor_info(tensor):
    return repr(tensor.shape)[6:] + ' ' + repr(tensor.dtype)[6:] + '@' + str(tensor.device) + '\n' + old_repr(tensor)
torch.Tensor.__repr__ = tensor_info

devices = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Alembic_Layer(nn.Module):
    def __init__(self, args, num_channels=32, signal_length=500, fs=128, numtaps=101):
        super(Alembic_Layer, self).__init__()
        torch.manual_seed(8989)

        self.fs = fs
        self.numtaps = numtaps
        self.num_channels = num_channels
        self.signal_length = signal_length

    def forward(self, x, filter_params_batch):
        batch_size = x.shape[0]//self.num_channels

        lowcuts, highcuts = filter_params_batch[:, :, 0], filter_params_batch[:, :, 1]

        filter_kernels = self.batch_bandpass_filter_kernel_vectorized(lowcuts, highcuts, self.fs, self.numtaps)

        x = x.view(-1, self.num_channels, self.signal_length)

        x = self.dynamic_conv(x, filter_kernels, batch_size)

        x = x.view(batch_size, 3, self.num_channels, self.signal_length)
        return x

    def dynamic_conv(self, x, filter_kernels, batch_size):
        in_channels, signal_length = x.size(1), x.size(2)
        num_filters = filter_kernels.size(1)  # Should be 3 for this example
        kernel_size = filter_kernels.size(2)

        filter_kernels = filter_kernels.view(batch_size * num_filters, 1, kernel_size)
        filter_kernels = filter_kernels.repeat(1, in_channels, 1)
        filter_kernels = filter_kernels.view(batch_size * in_channels * num_filters, 1, kernel_size)

        x = x.view(1, batch_size * in_channels, signal_length)
        x = nn.functional.conv1d(x, filter_kernels, groups=batch_size * in_channels, padding='same')
        x = x.view(batch_size, in_channels * num_filters, signal_length)
        return x

    def batch_bandpass_filter_kernel_vectorized(self, lowcuts, highcuts, fs, numtaps):
        batch_size = lowcuts.size(0)
        num_filters = lowcuts.size(1)
        nyquist = 0.5 * fs
        lows = lowcuts / nyquist
        highs = highcuts / nyquist
        n = torch.arange(numtaps,device=devices) - (numtaps - 1) / 2
        n = n.view(1, 1, -1).repeat(batch_size, num_filters, 1)  

        taps = (torch.sin(np.pi * n * highs.unsqueeze(2)) - torch.sin(np.pi * n * lows.unsqueeze(2))) / (np.pi * n)
        taps[:, :, (numtaps - 1) // 2] = highs - lows  

        window = torch.hann_window(numtaps).view(1, 1, -1).repeat(batch_size, num_filters, 1).to(devices)
        taps *= window

        return taps.view(batch_size, num_filters, numtaps)  

class Dynamic_Anchor_Layer(nn.Module): 
    def __init__(self, kernel_size=3, sigma=2, n_peaks = 3 ,min_peak_distance=3,
                 num_channels=32, signal_length=500, fs=128):
        super(Dynamic_Anchor_Layer, self).__init__()
        torch.manual_seed(8989)
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.sample_rate = fs
        self.signal_length = signal_length
        self.num_channels = num_channels
        self.min_peak_distance = min_peak_distance
        self.n_peaks = n_peaks

    def forward(self,x):
        x = x.view(-1,self.num_channels,self.signal_length)
        x, f_= self.analyze_signal_frequency(signal= x)
        x, _ = torch.sort(x,dim=-1)
        Cut_Line_1stStage = (x[:,1] - x[:,0])/2 + x[:,0]
        Cut_Line_2rdStage = (x[:,2] - x[:,1])/2 + x[:,1]

        LOW_lowcut = ((torch.ones(size=[x.shape[0]]))*0.5).unsqueeze(1).unsqueeze(1).to(devices)
        LOW_highcut = Cut_Line_1stStage.unsqueeze(1).unsqueeze(1)
        MID_lowcut = Cut_Line_1stStage.unsqueeze(1).unsqueeze(1)
        MID_highcut = Cut_Line_2rdStage.unsqueeze(1).unsqueeze(1)
        HIG_lowcut = Cut_Line_2rdStage.unsqueeze(1).unsqueeze(1)
        HIG_highcut = ((torch.ones(size=[x.shape[0]]))*45).unsqueeze(1).unsqueeze(1).to(devices)

        x = (torch.cat((LOW_lowcut,LOW_highcut,
                                MID_lowcut,MID_highcut,
                                HIG_lowcut,HIG_highcut),dim=1)).view(x.shape[0],3,2)
        return x, f_
    def gaussian_kernel(self):

        k = torch.arange(-self.kernel_size // 2 + 1, self.kernel_size // 2 + 1, dtype=torch.float32,device=devices)
        kernel = torch.exp(-k**2 / (2 * self.sigma**2))
        return kernel / kernel.sum()


    def analyze_signal_frequency(self, signal):
        batch_size,_ , signal_length = signal.shape
        fft_result = fft.rfft(signal)
        frequencies = fft.rfftfreq(signal_length, 1/self.sample_rate).to(devices)
        f_d= torch.abs(fft_result)**2
        f_ = f_d.mean(1)
        f_ = f_[:, 1:(signal_length//2)]
        frequencies = frequencies[1:(signal_length//2)]
        kernel = self.gaussian_kernel().to(devices)
        smoothed_psd = F.conv1d(f_.unsqueeze(1), kernel.view(1, 1, -1), padding=self.kernel_size//2).squeeze(1)
        diff = torch.diff(smoothed_psd, dim=1)
        peaks = (diff[:, :-1] > 0) & (diff[:, 1:] < 0)
        peaks = F.pad(peaks, (1, 1), "constant", 0)  
        peak_frequencies = torch.zeros((batch_size, self.n_peaks), dtype=torch.float32, device=devices)
        peak_magnitudes = torch.zeros((batch_size, self.n_peaks), dtype=torch.float32, device=devices)
        _, sorted_indices = torch.sort(smoothed_psd, dim=1, descending=True)
        sorted_peaks = torch.gather(peaks, 1, sorted_indices)
        sorted_frequencies = torch.gather(frequencies.unsqueeze(0).expand(batch_size, -1), 1, sorted_indices) 
        mask = torch.ones_like(sorted_peaks, dtype=torch.bool)
        for i in range(1, sorted_peaks.size(1)):
            mask[:, i] = torch.all(torch.abs(sorted_frequencies[:, i].unsqueeze(1) - sorted_frequencies[:, :i]) > self.min_peak_distance, dim=1)
        
        valid_peaks = sorted_peaks & mask
        cumsum = torch.cumsum(valid_peaks.float(), dim=1)
        top_n_indices = (cumsum <= self.n_peaks) & valid_peaks

        for i in range(batch_size):
            selected_indices = sorted_indices[i][top_n_indices[i]]
            num_selected = selected_indices.size(0)
            
            if num_selected > 0:
                peak_frequencies[i, :num_selected] = frequencies[selected_indices]
                peak_magnitudes[i, :num_selected] = smoothed_psd[i, selected_indices]
            
            # Fill remaining slots with random frequencies if needed
            if num_selected <self.n_peaks:
                remaining_slots = self.n_peaks - num_selected
                random_indices = torch.randint(0, len(frequencies), (remaining_slots,), device=devices)
                peak_frequencies[i, num_selected:] = frequencies[random_indices]
                peak_magnitudes[i, num_selected:] = smoothed_psd[i, random_indices]
        
        return peak_frequencies, f_d


class ACCNet(nn.Module):

    def __init__(self, args, in_channels, hidden_channels, out_channels):
        super(ACCNet, self).__init__()
        torch.manual_seed(9898)
        self.edge_compute = args.edge_compute
        self.batch_size = args.batch_size
        # hidden_channels = 16# heads = 4       
        in_channels = 500
        in_channels_psd = 251
        hidden_channels = args.GNN_inchans
        heads = args.GNN_inheads

        self.GAT_raw = GATv2Conv(in_channels_psd, hidden_channels,edge_dim=1, heads=heads, concat = True)
        self.GAT1 = GATv2Conv(in_channels, hidden_channels,edge_dim=1, heads=heads, concat = True)
        self.GAT2 = GATv2Conv(in_channels, hidden_channels,edge_dim=1, heads=heads, concat = True)
        self.GAT3 = GATv2Conv(in_channels, hidden_channels,edge_dim=1, heads=heads, concat = True)

        # self.BN_t = nn.BatchNorm1d(hidden_channels*12)

        self.encoder = nn.Sequential(#nn.Linear(384,64),
                                    #  nn.ReLU(),
                                      nn.BatchNorm1d(128),
                                     nn.Linear(128,2))

        self.AL = Alembic_Layer(args,fs=250)
        self.DAL = Dynamic_Anchor_Layer(fs=125)
        self.Cross = EdgeWeightingModel(num_edges=1024, channels=32) 
        # self.Beta = torch.tensor(0.5, dtype=torch.float32)
        self.Beta = args.T

    def forward(self, data):
        x, edge_index, Batch= data.x, data.edge_index, data.batch

        ## -------Adaptive Parameter Learning Unit (APU)---------##
        filter_params, f_d = self.DAL(x)
        ## -------Adaptive Dynamic Filtering Unit (DFU)----------##
        x = self.AL(x, filter_params)
        bz = x.shape[0]
        wave = x.shape[1]
        chans = x.shape[2]
        fea = x.shape[3]

        ##------------------Graph Initialization Builds-------------------##

        f_d= f_d.view(f_d.shape[0]*f_d.shape[1],f_d.shape[2])
        x_branch1 = x[:,0,:,:].reshape(x.shape[0]*x.shape[2],-1).float()
        x_branch2 = x[:,1,:,:].reshape(x.shape[0]*x.shape[2],-1).float()
        x_branch3 = x[:,2,:,:].reshape(x.shape[0]*x.shape[2],-1).float()
        

        edge_weight1 = ut.get_edge_attr(type=self.edge_compute, 
                                        signal_patch=(x[:,0,:,:]).reshape(x.shape[0]*x.shape[2],-1).float(), 
                                        batch=len(data.y)).cuda()
        edge_weight2 = ut.get_edge_attr(type=self.edge_compute, 
                                        signal_patch=(x[:,1,:,:]).reshape(x.shape[0]*x.shape[2],-1).float(), 
                                        batch=len(data.y)).cuda()
        edge_weight3 = ut.get_edge_attr(type=self.edge_compute, 
                                        signal_patch=(x[:,2,:,:]).reshape(x.shape[0]*x.shape[2],-1).float(), 
                                        batch=len(data.y)).cuda()
              
        f_weight = ut.get_edge_attr(type=self.edge_compute, signal_patch=f_d, 
                                        batch=len(data.y)).cuda()
        
        ##-------------------Cross-frequency Coupling Block (CFC)---------##

        _, edge_weight2, edge_weight3 = self.Cross(edge_weight1,edge_weight2,edge_weight3)


        x_psd = self.GAT_raw(f_d, edge_index, f_weight)
        x_psd = global_mean_pool(x_psd, Batch)

        x_branch1 = self.GAT1(x_branch1, edge_index, edge_weight1) #2048.128
        x_branch1 = global_mean_pool(x_branch1, Batch)

        x_branch2 = self.GAT2(x_branch2, edge_index, edge_weight2) #2048.128
        x_branch2 = global_mean_pool(x_branch2, Batch)

        x_branch3 = self.GAT3(x_branch3, edge_index, edge_weight3) #2048.128
        x_branch3 = global_mean_pool(x_branch3, Batch)

        # combined_branch = self.BN_t(combined_branch)
    
        combined_branch = x_psd + x_branch1 + self.Beta*(x_branch2 +x_branch3)

        x = self.encoder(combined_branch)

        return x



class EdgeWeightingModel(nn.Module):
    def __init__(self, num_edges, channels):
        super(EdgeWeightingModel, self).__init__()
        self.ch = channels
        self.weight_matrix = nn.Parameter(torch.ones(32, 32))
        self.alpha = nn.Parameter(torch.tensor(0.5))
        nn.init.kaiming_uniform_(self.weight_matrix, a=0.2, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, edges_1, edges_2, edges_3):



        edges_1 = edges_1.view(-1,self.ch, self.ch)
        edges_2 = edges_2.view(-1,self.ch, self.ch)
        edges_3 = edges_3.view(-1,self.ch, self.ch)
        

        weight_matrix = self.weight_matrix.view(32, 32)
        normalized_weight = nn.functional.softmax(self.weight_matrix.view(-1), dim=0).view(32, 32)
        

        weighted_edges_1 = edges_1 * normalized_weight.unsqueeze(0)
        

        edges_2_weighted = self.alpha * torch.bmm(weighted_edges_1, edges_2.transpose(1, 2)) + (1 - self.alpha) * edges_2
        edges_3_weighted = self.alpha * torch.bmm(weighted_edges_1, edges_3.transpose(1, 2)) + (1 - self.alpha) * edges_3

        edges_2_weighted = edges_2_weighted.view(-1)
        edges_3_weighted = edges_3_weighted.view(-1)
        
        return edges_1, edges_2_weighted, edges_3_weighted




    
class Output_encoder(nn.Module):

    def __init__(self):
        super(Output_encoder, self).__init__()
        self.mid_channels = 64
        self.final_out_channels = 128
        self.features_len = 1
        # model configs
        self.input_channels = 1
        self.kernel_size = 25
        self.stride = 6
        self.dropout = 0.2

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(self.input_channels, self.mid_channels, kernel_size=self.kernel_size,
                      stride=self.stride, bias=False, padding=(self.kernel_size // 2)),
            nn.BatchNorm1d(self.mid_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(self.dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(self.mid_channels, self.mid_channels * 2, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(self.mid_channels * 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(self.mid_channels * 2, self.final_out_channels, kernel_size=8, stride=1, bias=False,
                      padding=4),
            nn.BatchNorm1d(self.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        self.adaptive_pool = nn.AdaptiveAvgPool1d(self.features_len)

    def forward(self, x_in):
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.adaptive_pool(x)

        x_flat = x.reshape(x.shape[0], -1)
        return x_flat