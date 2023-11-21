"""Defines custom DataSet for processing the combination of the pedestrian action datasets JAAD and PIE:
Loading and Augmenting the data, and print statistics of the data
"""
from email.mime import image
import torch
import torch.utils.data as data
import os
import pickle5 as pk
from pathlib import Path
import numpy as np
from torchvision import transforms as A

from tqdm import tqdm, trange
from pie_data import PIE

# from jaad_data import JAAD
import sys
import numpy


# Define custom dataset class (crossing-behaviour prediction of pedestrians), data_set = (train, test, val) and data_type = (JAAD, PIE)
class DataSet(data.Dataset):
    def __init__(
        self,
        path,
        data_set,
        data_type,
        seq_type="crossing",
        sample_type="beh",
        balance=False,
        velocity=False,
        bbox=False,
    ):
        # Define random seed
        np.random.seed(42)
        # Specify dataset parameters (from inputs):
        # data_set is either (train, test, val)
        self.data_set = data_set
        # data_type is either (JAAD or PIE)
        self.data_type = data_type
        # seq_type is allways 'crossing' but the implementation of 'intention' and/or 'trajectory' prediction would be possible
        self.seq_type = seq_type
        # "beh" or "all" of the JAAD dataset (here allways "beh" will be used -> bystanders excluded)
        self.sample_type = sample_type
        # Balancing the classes (crossing, non-crossing)?
        self.balance = balance
        # Use bbox information?
        self.bbox = bbox
        # Use ego-vehicle velocity information?
        self.velocity = velocity
        # maximal augmentation of the pose keypoint and Bbox location (w = width, h = height)
        self.maxw_var = 8
        self.maxh_var = 6

        # Number of samples in dataset for different datasets and splits
        if (data_set == "train") and (data_type == "pie"):
            nsamples = [3635, 1272]
        elif (data_set == "train") and (data_type == "jaad"):
            nsamples = [400, 1926]
        elif (data_set == "test") and (data_type == "pie"):
            nsamples = [940, 366]
        elif (data_set == "test") and (data_type == "jaad"):
            nsamples = [133, 230]
        elif (data_set == "val") and (data_type == "pie"):
            nsamples = [358, 95]
        elif (data_set == "val") and (data_type == "jaad"):
            nsamples = [12, 33]

        # Calculate balance factors for each class
        balance_data = [max(nsamples) / s for s in nsamples]

        # Set data path based on the dataset and datasplit
        if (data_set == "train") and (data_type == "jaad"):
            self.data_path = os.getcwd() / Path(path) / "data/jaad/train/"
        elif (data_set == "train") and (data_type == "pie"):
            self.data_path = os.getcwd() / Path(path) / "data/pie/train/"
        elif (data_set == "test") and (data_type == "jaad"):
            self.data_path = os.getcwd() / Path(path) / "data/jaad/test/"
        elif (data_set == "test") and (data_type == "pie"):
            self.data_path = os.getcwd() / Path(path) / "data/pie/test/"
        elif (data_set == "val") and (data_type == "jaad"):
            self.data_path = os.getcwd() / Path(path) / "data/jaad/val"
        elif (data_set == "val") and (data_type == "pie"):
            self.data_path = os.getcwd() / Path(path) / "data/pie/val"

        # Get a list of data files in the specified path
        self.data_list = [data_name for data_name in os.listdir(self.data_path)]

        # Initialize data dictionary
        self.data = {}

        # Load and preprocess data
        for i in self.data_list:
            if (data_set == "train") and (data_type == "jaad"):
                p = "data/jaad/train/" + i
            elif (data_set == "train") and (data_type == "pie"):
                p = "data/pie/train/" + i
            elif (data_set == "test") and (data_type == "jaad"):
                p = "data/jaad/test/" + i
            elif (data_set == "test") and (data_type == "pie"):
                p = "data/pie/test/" + i
            elif (data_set == "val") and (data_type == "jaad"):
                p = "data/jaad/val/" + i
            elif (data_set == "val") and (data_type == "pie"):
                p = "data/pie/val/" + i

            loaded_data = self.load_data(p)
            if self.balance:
                if loaded_data["activities"] == 0:
                    self.repet_data(balance_data[0], loaded_data, i)
                elif loaded_data["activities"] == 1:
                    self.repet_data(balance_data[1], loaded_data, i)
            else:
                self.data[i.split(".")[0]] = loaded_data

        self.ped_ids = list(self.data.keys())
        self.data_len = len(self.ped_ids)

    # Method for replicating data to balance classes
    def repet_data(self, n_rep, data, ped_id):
        ped_id = ped_id.split(".")[0]

        prov = n_rep % 1
        n_rep = (
            int(n_rep)
            if prov == 0
            else int(n_rep) + np.random.choice(2, 1, p=[1 - prov, prov])[0]
        )

        for i in range(int(n_rep)):
            self.data[ped_id + f"-r{i}"] = data
            # print(ped_id + f'-r{i}'

    # Method for loading data from a file
    def load_data(self, data_path):
        with open(data_path, "rb") as fid:
            database = pk.load(fid, encoding="bytes")
        return database

    # Method to get the length of the dataset
    def __len__(self):
        return self.data_len

    # Method to get an item from the dataset
    def __getitem__(self, item):
        ped_id = self.ped_ids[item]
        data = self.data[ped_id]
        width = data["width"]
        height = width * 0.5626

        kp = data["pose"]
        bh = torch.from_numpy(np.array([data["activities"]])).float()
        # Augmentation only for training data
        if self.data_set == "train":
            kp[..., 0] = np.clip(
                kp[..., 0] + np.random.randint(self.maxw_var, size=kp[..., 0].shape),
                0,
                width,
            )
            kp[..., 1] = np.clip(
                kp[..., 1] + np.random.randint(self.maxh_var, size=kp[..., 1].shape),
                0,
                height,
            )

        # Normalization of keypoint location with image width/height to values [0, 1]
        kp[..., 0] /= width
        kp[..., 1] /= height
        # Return data with or without bounding box information
        if self.bbox == True:
            bbox = data["bbox"]
            # Normalization of bbox with image width/height to values [0, 1]
            bbox[..., 0] /= width
            bbox[..., 1] /= height
        # Return data with or without ego-vehicle velocity information
        if (self.velocity == True) and (self.data_type == "pie"):
            vel = data["vel"]

        if self.bbox == True:
            if (self.velocity == True) and (self.data_type == "pie"):
                return kp, bh, bbox, vel
            else:
                return kp, bh, bbox
        else:
            if (self.velocity == True) and (self.data_type == "pie"):
                return kp, bh, vel
            else:
                return kp, bh


# Main function for testing the dataset
def main():
    data_path = "./"
    train_data = []
    bbox = True
    numpy.set_printoptions(threshold=sys.maxsize)

    tr_data_jaad = DataSet(
        path=data_path,
        data_type="jaad",
        data_set="train",
        balance=True,
        bbox=bbox,
        velocity=False,
    )
    iter_ = tqdm(range(len(tr_data_jaad)))
    labels = np.zeros([len(tr_data_jaad), 2])

    # Iterate through the dataset and collect labels (and items)
    for i in iter_:
        if bbox == True:
            x, y, bb = tr_data_jaad.__getitem__(i)
        else:
            x, y = tr_data_jaad.__getitem__(i)
        labels[i, y.long().item()] = 1
    # Add JAAD data to the training data
    train_data.append(tr_data_jaad)
    print(f"No Crossing: {int(labels.sum(0)[0])}, Crossing: {int(labels.sum(0)[1])}")
    tr_data_pie = DataSet(
        path=data_path,
        data_type="pie",
        data_set="train",
        balance=True,
        bbox=bbox,
        velocity=False,
    )
    iter_ = tqdm(range(len(tr_data_pie)))
    labels = np.zeros([len(tr_data_pie), 2])

    # Iterate through the dataset and collect labels (and items)
    for i in iter_:
        if bbox == True:
            x, y, bb = tr_data_pie.__getitem__(i)
        else:
            x, y = tr_data_pie.__getitem__(i)
        labels[i + tr_data.__len__(), y_aug.long().item()] = 1

    print(f"No Crossing: {int(labels.sum(0)[0])}, Crossing: {int(labels.sum(0)[1])}")
    # Add PIE data to the training data
    train_data.append(tr_data_pie)
    tr_data = torch.utils.data.ConcatDataset(train_data)
    print(tqdm(range(len(tr_data))))

    print(bb)
    print("finish")


# Execute main function if script is run!
if __name__ == "__main__":
    main()
