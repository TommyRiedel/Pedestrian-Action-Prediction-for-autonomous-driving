"""Defines a model for binary crossing behaviour prediction of pedestrians (inputs: pose keypoint, bounding box location and ego vehicle velocity) 
Spatial Graph convolution and Temporal Convolution are combined to a ST-GCN which is applied to the pose and BBox location
The ego-vehcile velocity is processed with Temporal convolution layers.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torchsummary import summary
import pickle
import os
from pie_dataloader import DataSet
from torch.utils.data import DataLoader


# Definition of the main model class
class myModel(nn.Module):
    def __init__(self, nodes=17, n_clss=2, bbox=False, vel=False):
        super().__init__()

        # Initialization of model parameters
        # Number of classes (binary => 2)
        self.n_clss = n_clss
        # Number of nodes (pose) = 17 for HRNet
        self.nodes = nodes
        # Using bbox and ego vehicle velocity?
        self.bbox = bbox
        self.vel = vel

        # Definition of input channels (2 = x and y coordinate (human pose keypoints))
        # and convolutional layer feature size
        self.ch = 2
        self.ch1 = 32
        self.ch2 = 64

        # Batch normalization layers for input normalization of keypoint and bbox location(stabalizes + speeds up the training! 1D input data)
        # and regularization of the model
        self.data_bn = nn.BatchNorm1d(self.ch * nodes)
        bn_init(self.data_bn, 1)
        self.data_bn_b = nn.BatchNorm1d(self.ch * 2)
        bn_init(self.data_bn_b, 1)

        # Dropout layer for regularization
        self.drop = nn.Dropout(0.25)

        # Spatial partitioning strategy: initially anatomic connection: A = sum(A0, A1, A2);
        # A0 = identity matrix = self connections A = (3, nodes, nodes)
        A = Graph(nodes=17).get_spatial_graph()
        # Spatial partitioning strategy B = (3, 2, 2), stack of 3 identity matrices
        B = np.stack([np.eye(2)] * 3, axis=0)

        # Temporal convolution 1 (ego-vehicle velcotiy) (16 -> 32 features)
        if vel:
            self.v0 = nn.Sequential(
                nn.Conv1d(16, self.ch1, 1, bias=True),
                nn.BatchNorm1d(self.ch1),
                nn.SiLU(),
            )

        # ST-GCN block 1 (bounding box location - 2 nodes) (2 -> 32 features)
        if bbox:
            self.b1 = TCN_GCN_unit(self.ch, self.ch1, B, residual=False)

        # ST-GCN block 1 (pose keypoint location - 17 nodes) (2 -> 32 features)
        self.l1 = TCN_GCN_unit(self.ch, self.ch1, A, residual=False)

        # Temporal convolution 2 (ego-vehicle velcotiy) (32 -> 32 features)
        if vel:
            self.v1 = nn.Sequential(
                nn.Conv1d(self.ch1, self.ch1, 1, bias=True),
                nn.BatchNorm1d(self.ch1),
                nn.SiLU(),
            )

        # ST-GCN block 2 (bounding box location - 2 nodes) (32 -> 64 features)
        if bbox:
            self.b2 = TCN_GCN_unit(self.ch1, self.ch2, B)

        # ST-GCN block 2 (pose keypoint location - 17 nodes) (32 -> 64 features)
        self.l2 = TCN_GCN_unit(self.ch1, self.ch2, A)

        # Temporal convolution 3 (ego-vehicle velcotiy) (32 -> 64 features)
        if vel:
            self.v2 = nn.Sequential(
                nn.Conv1d(self.ch1, self.ch2, kernel_size=1, bias=True),
                nn.BatchNorm1d(self.ch2),
                nn.SiLU(),
            )

        # Adaptive AvgPool allows to specify desired output size regardless of input size! (dynamically adapts pooling) --> 1x1 output
        self.gap = nn.AdaptiveAvgPool2d(1)
        # Attention Block (weights features according to their expressiveness for the binary classification)
        self.att = nn.Sequential(
            nn.PReLU(),
            nn.Linear(self.ch2, self.ch2, bias=False),
            nn.BatchNorm1d(self.ch2),
            nn.Sigmoid(),
        )
        # Two Linear layers (64 -> 16 -> 2) + initialization
        self.linear1 = nn.Linear(self.ch2, 16)
        self.linear2 = nn.Linear(16, self.n_clss)
        nn.init.normal_(self.linear1.weight, 0, math.sqrt(2.0 / self.n_clss))
        if self.vel == False:
            nn.init.normal_(self.linear2.weight, 0, math.sqrt(2.0 / self.n_clss))

        # Pooling and fusion layers for feature fusion
        # Fusion of pose keypoint location and ego-vehicle velcotiy
        if vel:
            self.pool_sigm_1d = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Sigmoid())
        # Fusion of pose keypoint and bounding box location
        self.pool_sigm_2d = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Sigmoid())

    # Forward pass method for the model
    def forward(self, kp, bb, vel):
        ## comment out/in following (all kp =) if pose keypoints are not considered
        kp = kp.float()
        N, T, V, C = kp.shape
        kp = kp.permute(0, 3, 2, 1).contiguous().view(N, C * V, T)
        kp = self.data_bn(kp)
        kp = kp.view(N, C, V, T).permute(0, 1, 3, 2).contiguous()
        if self.vel:
            vel = vel.float()
            v1 = self.v0(vel)
        # comment out if pose keypoints are not considered
        x1 = self.l1(kp)
        if self.bbox:
            bb = bb.float()
            Nb, Tb, Vb, Cb = bb.shape
            bb = bb.permute(0, 3, 2, 1).contiguous().view(Nb, Cb * Vb, Tb)
            bb = self.data_bn_b(bb)
            bb = bb.view(Nb, Cb, Vb, Tb).permute(0, 1, 3, 2).contiguous()
            b1 = self.b1(bb)
            # comment out if pose keypoints are not considered
            x1.mul(self.pool_sigm_2d(b1))
        if self.vel:
            v1 = self.v1(v1)
            # comment out if pose keypoints are not considered
            x1 = x1.mul(self.pool_sigm_1d(v1).unsqueeze(-1))
        # comment out if pose keypoints are not considered
        x1 = self.l2(x1)
        if self.bbox:
            b1 = self.b2(b1)
            # comment out if pose keypoints are not considered
            x1.mul(self.pool_sigm_2d(b1))
            # comment in if pose keypoints are not considered
            # x1 = b1
        if self.vel:
            v1 = self.v2(v1)
            # comment out if pose keypoints are not considered
            x1 = x1.mul(self.pool_sigm_1d(v1).unsqueeze(-1))
            # comment in if pose keypoints and bbox are not considered
            # x1 = v1
        # comment out if ego-vehicle velocity is only input!!!
        x1 = self.gap(x1).squeeze(-1)

        x1 = x1.squeeze(-1)
        x1 = self.att(x1).mul(x1) + x1
        x1 = self.drop(x1)
        x1 = self.linear1(x1)
        x1 = self.linear2(x1)
        return x1


# Defines the spatial graph structure for graph convolutional networks = (3,17,17)
class Graph:
    def __init__(self, nodes=17):
        self.num_node = nodes
        # Self connections
        self.self_link = [(i, i) for i in range(self.num_node)]
        # Natural connection of human joints
        self.skeleton = [
            (1, 3),
            (1, 0),
            (2, 4),
            (2, 0),
            (0, 5),
            (0, 6),
            (5, 7),
            (7, 9),
            (6, 8),
            (8, 10),
            (5, 11),
            (6, 12),
            (11, 12),
            (11, 13),
            (13, 15),
            (12, 14),
            (14, 16),
        ]
        self.inward = [(i, j) for (i, j) in self.skeleton]
        self.outward = [(j, i) for (i, j) in self.inward]
        # all connections (self and to other nodes)
        self.neighbor = self.inward + self.outward

        self.A = self.get_spatial_graph()

    def get_spatial_graph(self):
        # Generates adjacency matrices for self-link, inward, and outward connections
        # I = self links, In = inward, out = outward
        I = self.edge2mat(self.self_link)
        In = self.normalize_digraph(self.edge2mat(self.inward))
        Out = self.normalize_digraph(self.edge2mat(self.outward))
        # Stacks these matrices to form the spatial graph tensor
        A = np.stack((I, In, Out))
        return A

    def edge2mat(self, link):
        # Converts edge connections to an adjacency matrix
        A = np.zeros((self.num_node, self.num_node))
        for i, j in link:
            A[j, i] = 1
        return A

    def normalize_digraph(self, A):
        # Normalizes the directed graph adjacency matrix
        Dl = np.sum(A, 0)
        h, w = A.shape
        Dn = np.zeros((w, w))
        for i in range(w):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i] ** (-1)
        AD = np.dot(A, Dn)
        return AD


# Initialization functions for convolutional layers
def conv_init(conv):
    # Initializes convolutional layer weights using Kaiming normal initialization
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode="fan_out")
    # Initializes bias term to zero if it exists
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    # Initializes batch normalization layer weights to the specified scale
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def conv_branch_init(conv, branches):
    # Initializes weights for convolutional layers in a branching structure
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2.0 / (n * k1 * k2 * branches)))
    # Initializes bias term to zero if it exists
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


# Definition of a temporal convolution unit
class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1):
        super().__init__()
        # Calculates the padding size for the temporal kernel
        pad = int((kernel_size - 1) / 2)

        # Definition of the temporal convolution, batch normalization, PReLU activation and Dropout (regularization) + initialization
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.prelu = nn.PReLU()
        self.dropout = nn.Dropout(0.3, inplace=True)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.prelu(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.dropout(x)
        return x


# Definition of a graph convolution unit
class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, adaptive=False):
        super().__init__()
        self.out_c = out_channels
        self.in_c = in_channels
        self.num_subset = A.shape[0]

        self.adaptive = adaptive
        if adaptive:
            # Adaptive adjacency matrix (trained)
            self.PA = nn.Parameter(
                torch.from_numpy(A.astype(np.float32)), requires_grad=True
            )
        else:
            # Fixed adjacency matrix
            self.A = torch.autograd.Variable(
                torch.from_numpy(A.astype(np.float32)), requires_grad=False
            )

        ## Convolutional layers used for each subset of the adjacency matrix
        self.conv_d = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        ## Downsampling block (Conv layer + batch norm.) - reduces number of channels if in_channels =/= out_channels
        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1), nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        # Batch Normalization and PReLU activation
        self.bn = nn.BatchNorm2d(out_channels)
        self.prelu = nn.PReLU()

        ## Initialization weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_subset):
            conv_branch_init(self.conv_d[i], self.num_subset)

    def L2_norm(self, A):
        # L2-normalization of the adjacency matrix
        A_norm = torch.norm(A, 2, dim=1, keepdim=True) + 1e-4  # N,1,V
        A = A / A_norm
        return A

    def forward(self, x):
        # N = batch_size, C = num. channels, T = temp. dim., v = num. graph nodes
        N, C, T, V = x.size()

        y = None
        if self.adaptive:
            A = self.PA
            A = self.L2_norm(A)
        else:
            A = self.A.cuda(x.get_device())
        ## for each subset of adjacency matrix - input tensor * corresponding subset of adjacency matrix -> passed through conv. layer ->
        for i in range(self.num_subset):
            A1 = A[i]
            A2 = x.view(N, C * T, V)
            z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
            y = z + y if y is not None else z
        y = self.prelu(self.bn(y) + self.down(x))

        return y


# Definition of a TCN-GCN unit
class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        super().__init__()
        # Definition of the GCN- und TCN-units
        self.gcn1 = unit_gcn(in_channels, out_channels, A)
        self.tcn1 = unit_tcn(out_channels, out_channels, stride=stride)
        self.prelu = nn.PReLU()
        # residual connection if activated
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        y = self.prelu(self.tcn1(self.gcn1(x)) + self.residual(x))
        return y


# Main function for initializing and running the model
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = myModel(nodes=17, n_clss=2, bbox=False, vel=False).to(device)


if __name__ == "__main__":
    main()
