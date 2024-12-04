# This is the processing script of DEAP dataset

import _pickle as cPickle

from train_model_eeg import *
from scipy import signal
from tqdm import tqdm

class PrepareData:
    def __init__(self, args):
        # init all the parameters here
        # arg contains parameter settings
        self.args = args
        self.data = None
        self.label = None
        self.model = None
        self.data_path = args.data_path
        self.label_type = args.label_type

    def run(self, subject_list, split=False, expand=True):

        data_combine = []
        label_combine = []
        pbar = tqdm(subject_list, colour = 'blue')


        for sub in pbar:


            data_= self.load_data_per_subject(sub)

            label_ = self.FACED_label_generator()

            if expand:

                data_ = np.expand_dims(data_, axis=-3)

            if split:
                data_, label_ = self.split(
                    data=data_, label=label_, segment_length=self.args.segment,
                    overlap=self.args.overlap, sampling_rate=self.args.sampling_rate)

            print('Data and label prepared!')
            print('data:' + str(data_.shape) + ' label:' + str(label_.shape))
            print('----------------------')
            self.save(data_, label_, sub)
            if self.args.data_using == 'FACED':

                data_combine.append(data_.reshape(-1,data_.shape[3],data_.shape[4]))
                label_combine.append(label_.reshape(-1))
                
        data_combine = np.array(data_combine)
        label_combine = np.array(label_combine)

        if not os.path.exists(self.args.combinedfeature_save_path):
            os.makedirs(self.args.combinedfeature_save_path)
        else:
            pass

        np.savez(self.args.combinedfeature_save_path+
                 self.args.label_type + '_Type_FACED_AllSub_Combined.npz'
                 , data=data_combine, label=label_combine)

        print("Save all the combined feature in ./feature")


    def FACED_label_generator(self):

        if self.label_type == 'NT':
            label = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]             
        
        if self.args.num_class == 2:
            print('Binary label generated!')
        return label


    def load_data_per_subject(self, sub):

        if (sub < 10):
            sub_code = str('sub00' + str(sub) + '.pkl')

        else:
            sub_code = str('sub0' + str(sub) + '.pkl')

        subject_path = os.path.join(self.data_path, sub_code)
        subject = cPickle.load(open(subject_path, 'rb'), encoding='latin1')

        if self.label_type == 'NT':
            data = np.delete(subject, slice(12, 16), axis=0)
        else:
            data = subject
        print('data:' + str(data.shape))
        sub += 1
        return data



    def save(self, data, label, sub):

        save_path = os.getcwd()
        data_type = 'data_{}_{}_{}'.format(self.args.data_format, self.args.dataset, self.args.label_type)
        save_path = osp.join(save_path, data_type)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        else:
            pass
        name = 'sub' + str(sub) + '.hdf'
        save_path = osp.join(save_path, name)
        dataset = h5py.File(save_path, 'w')
        dataset['data'] = data
        dataset['label'] = label
        dataset.close()

    def split(self, data, label, segment_length=1, overlap=0, sampling_rate=256):

        data_shape = data.shape
        step = int(segment_length * sampling_rate * (1 - overlap))
        data_segment = sampling_rate * segment_length
        data_split = []
        number_segment = int((data_shape[-1] - data_segment) // step)
        for i in range(number_segment + 1):
            data_split.append(data[:, :, :, (i * step):(i * step + data_segment)])
        data_split_array = np.stack(data_split, axis=1)
        label = np.stack([np.repeat(label[i], int(number_segment + 1)) for i in range(len(label))], axis=0)
        print("The data and label are split: Data shape:" + str(data_split_array.shape) + " Label:" + str(
            label.shape))
        data = data_split_array
        assert len(data) == len(label)
        return data, label
