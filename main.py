import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import coo_matrix
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
from sklearn.model_selection import train_test_split
import layers
criterion = torch.nn.CrossEntropyLoss()
import numpy as np
import argparse
from cross_validation import *
from prepare_data import *
os.chdir(sys.path[0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--notice', type=str, default='TRAINING ON FACED DATASET')
    parser.add_argument('--data-path', type=str, default='./data/raw_EEG/FACED_data_preprocessed_python')
    parser.add_argument('--combinedfeature-save-path', default='./data/features/') 
    parser.add_argument('--num-electrodes', type=int, default=32)
    parser.add_argument('--feature-length', type=int, default=500, help="FACED:500")
    parser.add_argument('--label-type', type=str, default='NT', choices=['A', 'V', 'D', 'L',
                                                                         'NT','N', 'T', 'P','ALL'], 
                                                                         help='NT means remove the Neutral')
    parser.add_argument('--data-prepare', type=bool, default=True)
    parser.add_argument('--data-using', type=str, default='FACED', choices=['FACED'])
    parser.add_argument('--dataset', type=str, default='FACED', choices=['FACED'])
    parser.add_argument('--subject_begin', type=int, default=0, help='If the programme is interrupted subject_begin can be quickly restarted')
    parser.add_argument('--subjects', type=int, default=40)
    parser.add_argument('--num-class', type=int, default=2, choices=[2])
    parser.add_argument('--segment', type=int, default=2)
    parser.add_argument('--overlap', type=float, default=0)
    parser.add_argument('--sampling-rate', type=int, default=250, help="FACED=250")
    parser.add_argument('--data-format', type=str, default='eeg')    
    parser.add_argument('--save-path', default='./results/')
    parser.add_argument('--pool', type=int, default=16)
    parser.add_argument('--model', type=str, default='ACCNet', choices=['ACCNet'])
    parser.add_argument('--dim-expand', type=bool, default=False, help="Expand data dim to adapte CNN-based model ")
    parser.add_argument('--edge-compute', type=str, default='COS', choices=['PLI','COS'])
    parser.add_argument('--load-path', default='./results/max-f1.pth')
    parser.add_argument('--load-path-final', default='./results/max-f1.pth', help="if reproduce, set final_model.pth")
    parser.add_argument('--save-model', type=bool, default=True)
    parser.add_argument('--data-norm', type=bool, default=True, help="normalize the raw data")
    parser.add_argument('--random-seed', type=int, default=666)
    parser.add_argument('--dense', type=str, default='full', choices=['full', 'sparse', 'phy'])
    parser.add_argument('--max-epoch', type=int, default=200, help="default max epoch 200")
    parser.add_argument('--early-stop-counter',type=int,default=70, help="early stop counter")
    parser.add_argument('--cts', type=bool, default=False, help="combine training switch")
    parser.add_argument('--patient-cmb', type=int, default=8)
    parser.add_argument('--max-epoch-cmb', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--step-size', type=int, default=5)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--hidden', type=int, default=32)
    parser.add_argument('--LS', type=bool, default=True, help="Label smoothing")
    parser.add_argument('--LS-rate', type=float, default=0.1)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--T', type=float, default=0.7, help="BeTa, control B2,B3")
    parser.add_argument('--GNN-inchans', type=int, default=32)
    parser.add_argument('--GNN-inheads', type=int, default=4)
    parser.add_argument('--reproduce', type= bool, default=False)

    args = parser.parse_args()

    if args.data_using == 'FACED':
        subject = np.arange(args.subject_begin, args.subjects)
        sub_to_run = subject

    ###########DATA PREPARE########## 
    if args.data_prepare:
        pd = PrepareData(args)
        pd.run(sub_to_run, split=True, expand=True)
        print("Data Prepared")
    #################################

    cv = CrossValidation(args)
    cv.n_fold_CV(subject=subject)

