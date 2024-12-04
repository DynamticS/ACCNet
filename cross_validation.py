import numpy as np
import datetime
import os
import csv
import h5py
import copy
import os.path as osp
from train_model_eeg import *
from utils import Averager, ensure_path
from sklearn.model_selection import KFold
import pickle
ROOT = os.getcwd()

####Declaration#####
## The cross-validation section is modified based on LGGNet. 
## If you wish to use it, please cite accordingly.
## Ding, Yi, et al. "LGGNet: Learning from local-global-graph representations for brain–computer interface." IEEE Transactions on Neural Networks and Learning Systems (2023).


class CrossValidation:
    def __init__(self, args):
        self.args = args
        self.data = None
        self.label = None
        self.model = args.model
        self.data_dir = args.data_path
        self.label_type = args.label_type
        self.data_using = args.data_using
        self.cts = args.cts
        self.data_norm = args.data_norm
        # Log the results per subject
        result_path = osp.join(args.save_path, 'result')
        ensure_path(result_path)
        self.text_file = osp.join(result_path,
                                  "results_{}.txt".format(args.data_using))
        file = open(self.text_file, 'a')

        file.write("\n" + str(datetime.datetime.now()) +
                   "\nTrain:Parameter setting for " + str(args.model) + ' on ' + str(args.data_using) +
                   "\n0)Notice:" + str(args.notice) +
                   "\n1)Label_type:" + str(args.label_type) +
                   "\n2)Random_seed:" + str(args.random_seed) +
                   "\n3)Learning_rate:" + str(args.learning_rate) +
                   "\n4)Esc:" + str(args.early_stop_counter) +
                   "\n5)Num_epochs:" + str(args.max_epoch) +
                   "\n6)Batch_size:" + str(args.batch_size) +
                   "\n7)Dropout:" + str(args.dropout) +
                   "\n8)Hidden_node:" + str(args.hidden) +
                   "\n9)Dense method:" + str(args.dense) +
                   "\n10)Max_epoch_combine_train:" + str(args.max_epoch_cmb) +
                   "\n11)T:" + str(args.T) + 
                   "\n12)Edge_compute_using:" + str(args.edge_compute) +
                   "\n13)GAT_heads:" + str(args.GNN_inheads) + '\n')
        file.close()

    def load_per_subject(self, sub):

        save_path = os.getcwd()
        data_type = 'data_{}_{}_{}'.format(self.args.data_format, self.args.dataset, self.args.label_type)
        sub_code = 'sub' + str(sub) + '.hdf'
        path = osp.join(save_path, data_type, sub_code)
        dataset = h5py.File(path, 'r')
        data = np.array(dataset['data'])
        label = np.array(dataset['label'])
        print('>>> Data:{} Label:{}'.format(data.shape, label.shape))
        return data, label


    def load_eeg_data(self):

        if self.data_using == 'FACED':

            dir = self.data_dir
            label_type = self.label_type
            dataset_ = np.load(self.args.combinedfeature_save_path+
                            self.args.label_type + '_Type_FACED_AllSub_Combined.npz')
            data_ = dataset_['data']
            lbls_ = dataset_['label']
            
        else:
            pass

        print(label_type + " labeltype shape:", data_.shape)
        print(label_type + " labeltype shape:", lbls_.shape)

        return data_, lbls_

    def normalize(self, train, test):

        for channel in range(train.shape[1]):
            mean = np.mean(train[:, channel, :])
            std = np.std(train[:, channel, :])
            train[:, channel, :] = (train[:, channel, :] - mean) / std
            test[:, channel, :] = (test[:, channel, :] - mean) / std
        return train, test

    def split_balance_class(self, data, label, train_rate, random):
        
        np.random.seed(0)

        index_0 = np.where(label == 0)[0]
        index_1 = np.where(label == 1)[0]

        # for class 0
        index_random_0 = copy.deepcopy(index_0)

        # for class 1
        index_random_1 = copy.deepcopy(index_1)

        if random == True:
            np.random.shuffle(index_random_0)
            np.random.shuffle(index_random_1)

        index_train = np.concatenate((index_random_0[:int(len(index_random_0) * train_rate)],
                                      index_random_1[:int(len(index_random_1) * train_rate)]),
                                     axis=0)
        index_val = np.concatenate((index_random_0[int(len(index_random_0) * train_rate):],
                                    index_random_1[int(len(index_random_1) * train_rate):]),
                                   axis=0)

        # get validation
        val = data[index_val]
        val_label = label[index_val]

        train = data[index_train]
        train_label = label[index_train]

        return train, train_label, val, val_label


    def n_fold_CV(self, subject, fold=10):

        # Train and evaluate the model subject by subject
        tta = []  # total test accuracy
        tva = []  # total validation accuracy
        ttf = []  # total test f1
        tvf = []  # total validation f1
        #subject: 0->40
        data, label = self.load_eeg_data()  #
        for sub in subject:

            data_ = data[int(sub)]
            label_ = label[int(sub)]
            va_val = Averager()
            vf_val = Averager()
            preds, acts = [], []
            # kf = KFold(n_splits=fold, shuffle=True, random_state=8989)
            kf = KFold(n_splits=fold, shuffle=True, random_state=8989)
            for idx_fold, (idx_train, idx_test) in enumerate(kf.split(data_)):
                print('Outer loop: {}-fold-CV Fold:{}'.format(fold, idx_fold))

                data_train = data_[idx_train]
                label_train = label_[idx_train]
                data_test = data_[idx_test]
                label_test = label_[idx_test]
                if self.data_norm == True:
                    data_train, data_test = self.normalize(train=data_train, test=data_test)
                # Change data to Tensor Format
                data_train = torch.from_numpy(data_train).float()
                label_train = torch.from_numpy(label_train).long()
                data_test = torch.from_numpy(data_test).float()
                label_test = torch.from_numpy(label_test).long()

                if self.args.reproduce:
                    # to reproduce the reported ACC
                    acc_test, pred, act = test(args=self.args, data=data_test, label=label_test,
                                               reproduce=self.args.reproduce,
                                               subject=sub, fold=idx_fold)
                    acc_val = 0
                    f1_val = 0
                else:
                    # to train new models
                    acc_val, f1_val = self.first_stage(data=data_train, label=label_train,
                                                       subject=sub, fold=idx_fold)

                    if self.cts:
                        combine_train(args=self.args,
                                      data=data_train, label=label_train,
                                      subject=sub, fold=idx_fold, target_acc=1)

                    acc_test, pred, act = test(args=self.args, data=data_test, label=label_test,
                                               reproduce=self.args.reproduce,
                                               subject=sub, fold=idx_fold)
                va_val.add(acc_val)
                vf_val.add(f1_val)
                preds.extend(pred)
                acts.extend(act)

            tva.append(va_val.item())
            tvf.append(vf_val.item())
            acc, f1, _ = get_metrics(y_pred=preds, y_true=acts)
            tta.append(acc)
            ttf.append(f1)
            result = '{},{}'.format(tta[-1], f1)
            self.log2txt(result)

        # prepare final report
        tta = np.array(tta) #total test acc
        ttf = np.array(ttf) #total test F1
        tva = np.array(tva) #total validation acc
        tvf = np.array(tvf) #total validation F1
        mACC = np.mean(tta)
        std = np.std(tta)
        mF1 = np.mean(ttf)
        F1_std = np.std(ttf)

        mACC_val = np.mean(tva)
        std_val = np.std(tva)
        mF1_val = np.mean(tvf)

        print('Final: test mean ACC:{} std:{}'.format(mACC, std))
        print('Final: val mean ACC:{} std:{}'.format(mACC_val, std_val))
        print('Final: test mean F1:{} std:{}'.format(mF1, F1_std))
        print('Final: val mean F1:{}'.format(mF1_val))
        results = 'test mAcc={} mAcc std={} mF1={} mF1 std={} val mAcc={} val mF1={}'.format(mACC,std,
        mF1,F1_std, mACC_val, mF1_val)
        self.log2txt(results)




    def first_stage(self, data, label, subject, fold):

        kf = KFold(n_splits=3, shuffle=True, random_state=8989)
        va = Averager()
        vf = Averager()
        va_item = []
        vf_item =[]
        maxAcc = 0.0
        maxF1 = 0.0
        for i, (idx_train, idx_val) in enumerate(kf.split(data)):
            print('Inner 3-fold-CV Fold:{}'.format(i))
            data_train, label_train = data[idx_train], label[idx_train]
            data_val, label_val = data[idx_val], label[idx_val]
            acc_val, F1_val, F1_max = train(args=self.args,
                                    data_train=data_train,
                                    label_train=label_train,
                                    data_val=data_val,
                                    label_val=label_val,
                                    subject=subject,
                                    fold=fold)
            va.add(acc_val)
            vf.add(F1_val)
            va_item.append(acc_val)
            vf_item.append(F1_max)

            if F1_max >= maxF1:
                maxF1 = F1_max
                # choose the model with higher val as the model to second stage
                old_name = osp.join(self.args.save_path, 'candidate.pth')
                new_name = osp.join(self.args.save_path, 'max-f1.pth')
                if os.path.exists(new_name):
                    os.remove(new_name)
                os.rename(old_name, new_name)
                print('New max F1 model saved, with the val F1 being:{}'.format(maxF1))

        mAcc = va.item()
        mF1 = vf.item()
        return mAcc, mF1

    def log2txt(self, content):
  
        file = open(self.text_file, 'a')
        file.write(str(content) + '\n')
        file.close()


