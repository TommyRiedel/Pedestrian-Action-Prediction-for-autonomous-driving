from email.mime import image
from re import S
import torch
import torch.utils.data as data
import os
import pickle5 as pk
from pathlib import Path
import numpy as np
from torchvision import transforms as A

from tqdm import tqdm, trange
from pie_data import PIE
import sys
import numpy



class DataSet(data.Dataset):
    def __init__(self, path, data_set, seq_type='crossing', sample_type='beh', balance=False, velocity=False, bbox=False):
        
        np.random.seed(42)
        # either 'train', 'test', or, 'val'
        self.data_set = data_set
        # either 'intention', 'trajectory', 'crossing'
        self.seq_type = seq_type
        # "beh" or "all"
        self.sample_type=sample_type
        self.balance = balance
        self.bbox = bbox
        self.velocity = velocity
        # augmentation
        self.maxw_var = 8
        self.maxh_var = 6

        ## Number of samples in dataset
        if data_set == 'train':
            nsamples = [3635, 1272]
        elif data_set == 'test':
            nsamples = [940, 366]
        elif data_set == 'val':
            nsamples = [358, 95]

        balance_data = [max(nsamples) / s for s in nsamples]

        if data_set == 'train':
            self.data_path = os.getcwd() / Path(path) / 'data/pie/train/'
        elif data_set == 'test':
            self.data_path = os.getcwd() / Path(path) / 'data/pie/test/'
        elif data_set == 'val':
            self.data_path = os.getcwd() / Path(path) / 'data/pie/val'
        self.data_list = [data_name for data_name in os.listdir(self.data_path)]
        
        self.data = {}

        for i in self.data_list:
            if data_set == 'train':
                p = 'data/pie/train/' + i
            elif data_set == 'test':
                p = 'data/pie/test/' + i
            elif data_set == 'val':
                p = 'data/pie/val/' + i
            
            loaded_data = self.load_data(p)
            ### ToDo: Nochmal checken!
            if self.balance:# and (data_set == 'test' or data_set == 'val'):
                if loaded_data['activities'] == 0:
                    self.repet_data(balance_data[0], loaded_data, i)
                elif loaded_data['activities'] == 1:
                    self.repet_data(balance_data[1], loaded_data, i)
            else:
                #print(i.split('.')[0])
                self.data[i.split('.')[0]] = loaded_data
        self.ped_ids = list(self.data.keys())
        self.data_len = len(self.ped_ids)
        
    def repet_data(self, n_rep, data, ped_id):
        ped_id = ped_id.split('.')[0]
        
        prov = n_rep % 1 
        n_rep = int(n_rep) if prov == 0 else int(n_rep) + np.random.choice(2, 1, p=[1 - prov, prov])[0]
        
        for i in range(int(n_rep)):
            self.data[ped_id + f'-r{i}'] = data
            #print(ped_id + f'-r{i}')

    def load_data(self, data_path):
        with open(data_path, 'rb') as fid:
            database = pk.load(fid, encoding='bytes')
        return database

    def __len__(self):
        return self.data_len
        
    def __getitem__(self, item):
        ped_id = self.ped_ids[item]
        data = self.data[ped_id]
        width = data['width']
        height = width * 0.5626

        kp = data['pose']
        bh = torch.from_numpy(np.array([data['activities']])).float()
        if self.data_set == 'train':
            # Augmentation
            kp[..., 0] = np.clip(kp[..., 0] + np.random.randint(self.maxw_var, size=kp[..., 0].shape), 0, width)
            kp[..., 1] = np.clip(kp[..., 1] + np.random.randint(self.maxh_var, size=kp[..., 1].shape), 0, height)

        # Normalization
        kp[..., 0] /= width
        kp[..., 1] /= height
        if self.bbox == True:
            bbox = data['bbox']
            # Normalization
            bbox[..., 0] /= width
            bbox[..., 1] /= height

        if self.velocity == True:
            vel = data['vel']
        if self.bbox == True:
            if self.velocity == True:
                return kp, bh, bbox, vel
            else:
                return kp, bh, bbox
        else:
            if self.velocity == True:
                return kp, bh, vel
            else:
                return kp, bh

def main():
    data_path = './'
    numpy.set_printoptions(threshold=sys.maxsize)
    velocity = True
    bbox = True
    #te_data = DataSet(path=data_path, data_set='test', balance=False, velocity=velocity)
    tr_data = DataSet(path=data_path, data_set='train', balance=False, velocity=velocity, bbox=bbox)
    iter_ = tqdm(range(len(tr_data)))
    labels = np.zeros([len(tr_data), 2])
    id = []

    for i in iter_:
        if (bbox == True) and (velocity == True):
            x, y, bb, vel = tr_data.__getitem__(i)
        elif (bbox == True) and (velocity == False):
            x, y, bb = tr_data.__getitem__(i)
        elif (bbox == False) and (velocity == True):
            x, y, vel = tr_data.__getitem__(i)
        else:
            x, y = tr_data.__getitem__(i)
        labels[i, y.long().item()] = 1
        #id.append(pid)
    #j = 0
    #for i in iter_:
    #    if labels[i, 1] == 1:
    #        print(id[i], labels[i])
    #        j += 1
    #print(j)
    print(bb)
    print(f'No Crossing: {int(labels.sum(0)[0])}, Crossing: {int(labels.sum(0)[1])}')
    print('finish')

if __name__ == "__main__":
    main()
            

