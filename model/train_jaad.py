"""
"""
import torch
from torchvision import transforms as A
from torch.utils.data import DataLoader
from torch.nn import functional as F

from torch import nn, optim
from tqdm import tqdm
from torch.utils.data import random_split

import pytorch_lightning as pl
import torchmetrics
from torchmetrics import Metric
from torchmetrics.functional.classification.accuracy import accuracy
from torchmetrics.functional import precision
from torchmetrics.functional import recall
from torchmetrics.functional.classification.confusion_matrix import (
    binary_confusion_matrix,
)
from torchmetrics.functional import f1_score
from torchmetrics.functional import auroc

from sklearn.metrics import balanced_accuracy_score
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import LearningRateFinder
from pytorch_lightning.loggers import TensorBoardLogger

from model import myModel
from jaad_dataloader import DataSet

from pathlib import Path
import argparse
import os
import numpy as np
import pickle
from torchsummary import summary

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from collections import OrderedDict


# Function to set seed for reproducibility
def seed_everything(seed):
    torch.cuda.empty_cache()
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


# Lightning Module for the model
class LitMyModel(pl.LightningModule):
    def __init__(self, args, len_tr):
        super().__init__()

        # Specification of model parameters pased on Argument Parser
        # Number of total steps = len training data times epoch size
        self.total_steps = len_tr * args.epochs
        self.lr = args.lr
        self.epochs = args.epochs
        self.balance = args.balance
        self.bbox = args.bbox
        self.velocity = args.velocity
        # Number of nodes of the human pose estimated via HRNet
        nodes = 17
        self.validation_step_preds = []
        self.validation_step_ys = []

        # Initialize model and set device
        self.model = myModel(nodes=nodes, n_clss=2, bbox=args.bbox, vel=False)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Number of Data-Sequences in each dataset (train, test, val) -> balance weights
        ## NC ## C
        # Set weights for data imbalance
        tr_nsamples = [400, 1926]
        self.tr_weight = (
            torch.from_numpy(np.min(tr_nsamples) / tr_nsamples).float().to(device)
        )
        print(self.tr_weight)
        val_nsamples = [12, 33]
        self.val_weight = (
            torch.from_numpy(np.min(val_nsamples) / val_nsamples).float().to(device)
        )
        print(self.val_weight)
        te_nsamples = [133, 230]
        self.te_weight = (
            torch.from_numpy(np.min(te_nsamples) / te_nsamples).float().to(device)
        )
        print(self.te_weight)

    def forward(self, kp, bb, vel):
        y = self.model(kp, bb, vel)
        return y

    def training_step(self, batch, batch_nb):
        # Training step logic: x = keypoint location, y = label, bb = BBox location, vel = veloctiy (not for JAAD)
        x = batch[0]
        y = batch[1]
        if self.bbox:
            bb = batch[2]
        else:
            bb = None
        vel = None

        logits = self(x, bb, vel)
        w = None if self.balance else self.tr_weight
        y_onehot = torch.FloatTensor(y.shape[0], 2).to(y.device).zero_()
        y_onehot.scatter_(1, y.long(), 1)
        loss = F.binary_cross_entropy_with_logits(logits, y_onehot, weight=w)

        preds = logits.softmax(1).argmax(1)
        acc = accuracy(preds.view(-1).long(), y.view(-1).long(), task="binary")
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc * 100.0, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_nb):
        # Validation step logic: x = keypoint location, y = label, bb = BBox location, vel = veloctiy (not for JAAD)
        x = batch[0]
        y = batch[1]
        if self.bbox:
            bb = batch[2]
        else:
            bb = None
        vel = None

        logits = self(x, bb, vel)
        w = None  # if self.balance else self.val_weight
        y_onehot = torch.FloatTensor(y.shape[0], 2).to(y.device).zero_()
        y_onehot.scatter_(1, y.long(), 1)
        loss = F.binary_cross_entropy_with_logits(logits, y_onehot, weight=w)

        preds = logits.softmax(1).argmax(1)
        acc = accuracy(preds.view(-1).long(), y.view(-1).long(), task="binary")
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc * 100.0, prog_bar=True)
        return loss

    def test_step(self, batch, batch_nb):
        # Test step logic: x = keypoint location, y = label, bb = BBox location, vel = veloctiy (not for JAAD)
        x = batch[0]
        y = batch[1]
        if self.bbox:
            bb = batch[2]
        else:
            bb = None
        vel = None

        logits = self(x, bb, vel)
        w = None  # if self.balance else self.test_weight
        y_onehot = torch.FloatTensor(y.shape[0], 2).to(y.device).zero_()
        y_onehot.scatter_(1, y.long(), 1)
        loss = F.binary_cross_entropy_with_logits(logits, y_onehot, weight=w)

        preds = logits.softmax(1).argmax(1)
        acc = accuracy(preds.view(-1).long(), y.view(-1).long(), task="binary")

        f1 = f1_score(preds.view(-1).long(), y.view(-1).long(), task="binary")
        AUROC = auroc(
            logits.softmax(1).max(1)[0].view(-1), y.view(-1).long(), task="binary"
        )
        pre = precision(
            preds.view(-1).long(), y.view(-1).long(), task="binary", average=None
        )
        re = recall(
            preds.view(-1).long(), y.view(-1).long(), task="binary", average=None
        )

        self.validation_step_preds.append(preds.view(-1).long())
        self.validation_step_ys.append(y.view(-1).long())

        # Log Loss, Accuracy, F1-score, AUROC, precision and recall
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc * 100.0, prog_bar=True)
        self.log("f1-score", f1 * 100.0, prog_bar=True)
        self.log("AUROC", AUROC, prog_bar=True)
        self.log("precision", pre * 100.0, prog_bar=True)
        self.log("recall", re * 100.0, prog_bar=True)
        return loss, preds

    def on_test_epoch_end(self):
        # Logic to calculate and log metrics at the end of the test epoch (create confusion matrix)
        y_hat = torch.cat(self.validation_step_preds)
        y = torch.cat(self.validation_step_ys)

        confmat = binary_confusion_matrix(y_hat, y, threshold=0.3)
        confmat_computed = confmat.detach().cpu().numpy().astype(int)
        df_cm = pd.DataFrame(confmat_computed)
        plt.figure(figsize=(10, 7))
        fig_ = sns.heatmap(df_cm, annot=True, fmt="d", cmap="Spectral").get_figure()
        plt.savefig("./confmat.png")
        plt.close(fig_)
        self.loggers[0].experiment.add_figure(
            "Confusion matrix", fig_, self.current_epoch
        )

    def configure_optimizers(self):
        # Configure optimizer (AdamW) and learning rate scheduler (OneCycleLR)
        optm = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-3)
        # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optm, gamma=0.1, last_epoch=-1, verbose=False)
        lr_scheduler = {
            "name": "OneCycleLR",
            "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                optm,
                max_lr=self.lr,
                div_factor=2.5,
                final_div_factor=1e5,
                total_steps=self.total_steps,
                verbose=False,
                pct_start=0.20,
            ),
            "interval": "step",
            "frequency": 1,
        }
        return [optm], [lr_scheduler]

    def predict_step(self, batch, batch_idx):
        # Prediction step logic: x = keypoint location, y = label, bb = BBox location, vel = veloctiy (not for JAAD)
        x = batch[0]
        y = batch[1]
        if self.bbox:
            bb = batch[2]
        else:
            bb = None
        vel = None

        return self(x, bb, vel).softmax(1), self(x, bb, vel).softmax(1).argmax(1), y


# Main function to run the training and testing
def main(args):
    seed_everything(args.seed)
    # Load train, validation and test DataSets (JAAD)
    tr_data = DataSet(
        path=args.data_path,
        data_set="train",
        balance=False,
        bbox=args.bbox,
        velocity=args.velocity,
    )
    te_data = DataSet(
        path=args.data_path,
        data_set="test",
        balance=False,
        bbox=args.bbox,
        velocity=args.velocity,
    )
    val_data = DataSet(
        path=args.data_path,
        data_set="val",
        balance=False,
        bbox=args.bbox,
        velocity=args.velocity,
    )

    # Create DataLoader for each dataset
    tr = DataLoader(
        tr_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    te = DataLoader(
        te_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # TensorBoard logger
    logger = TensorBoardLogger("tb_logs", name="final_jaad")
    mymodel = LitMyModel(args, len(tr))

    if not Path(args.logdir).is_dir():
        os.mkdir(args.logdir)

    # Model checkpointing and other callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.logdir,
        monitor="val_acc",
        save_top_k=5,
        filename="jaad-{epoch:02d}-{val_acc:.3f}",
        save_weights_only=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    lr_finder = LearningRateFinder(min_lr=1e-9, max_lr=0.1, early_stop_threshold=None)

    # Trainer configuration
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        min_epochs=1,
        max_epochs=args.epochs,
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor, lr_finder],
        precision=args.precision,
    )

    # Train the model
    trainer.fit(mymodel, tr, val)
    # Where store weigths of model? -> args.logdir + 'jaad_case_' + 'p' = pose, 'v' = ego-vehicle vel., 'b' = bbox + '.pth'
    torch.save(mymodel.model.state_dict(), args.logdir + "jaad_case_pb.pth")
    # Test the model
    trainer.test(mymodel, te)
    print("finish")


if __name__ == "__main__":
    torch.cuda.empty_cache()
    # Parse command line arguments
    parser = argparse.ArgumentParser("Pedestrian prediction crossing")
    # logger directory to store data for tensorboard
    parser.add_argument(
        "--logdir",
        type=str,
        default="./data/jaad/tb/",
        help="logger directory for tensorboard",
    )
    # specify number of epochs
    parser.add_argument(
        "--epochs", type=int, default=175, help="Number of eposch to train"
    )
    # specify (max) learning rate
    parser.add_argument("--lr", type=int, default=5e-3, help="learning rate to train")
    # path to directoy with data
    parser.add_argument(
        "--data_path", type=str, default="./", help="Path to the train and test data"
    )
    # specify batch size
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for training and test"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=20,
        help="Number of workers for the dataloader",
    )
    # If true: obd velocity of the ego vehicle will be used in the model
    parser.add_argument(
        "--velocity",
        type=bool,
        default=False,
        help="activate the use of the odb and gps velocity",
    )
    # If true: bounding box location will be used in the model
    parser.add_argument(
        "--bbox", type=bool, default=True, help="activate the use of the bounding box"
    )
    parser.add_argument(
        "--balance", type=bool, default=False, help="Balnce or not the data set"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--accelerator", type=str, default="gpu", help="GPU, TPU or CPU"
    )
    parser.add_argument("--devices", type=list, default=[0], help="Which GPU")
    parser.add_argument("--precision", type=int, default=16, help="precision of result")

    args = parser.parse_args()
    main(args)
