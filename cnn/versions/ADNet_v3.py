# numpy
# torch
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchsummary import summary

# matplotlib

filter_class_1 = [
    np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, 0]
    ], dtype=np.float32),
    np.array([
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 0]
    ], dtype=np.float32),
    np.array([
        [0, 0, 1],
        [0, -1, 0],
        [0, 0, 0]
    ], dtype=np.float32),
    np.array([
        [0, 0, 0],
        [1, -1, 0],
        [0, 0, 0]
    ], dtype=np.float32),
    np.array([
        [0, 0, 0],
        [0, -1, 1],
        [0, 0, 0]
    ], dtype=np.float32),
    np.array([
        [0, 0, 0],
        [0, -1, 0],
        [1, 0, 0]
    ], dtype=np.float32),
    np.array([
        [0, 0, 0],
        [0, -1, 0],
        [0, 1, 0]
    ], dtype=np.float32),
    np.array([
        [0, 0, 0],
        [0, -1, 0],
        [0, 0, 1]
    ], dtype=np.float32)
]

filter_class_2 = [
    np.array([
        [1, 0, 0],
        [0, -2, 0],
        [0, 0, 1]
    ], dtype=np.float32),
    np.array([
        [0, 1, 0],
        [0, -2, 0],
        [0, 1, 0]
    ], dtype=np.float32),
    np.array([
        [0, 0, 1],
        [0, -2, 0],
        [1, 0, 0]
    ], dtype=np.float32),
    np.array([
        [0, 0, 0],
        [1, -2, 1],
        [0, 0, 0]
    ], dtype=np.float32),
]

filter_class_3 = [
    np.array([
        [3, 0, 0],
        [0, -3, 0],
        [0, 0, 1],
    ], dtype=np.float32),
    np.array([
        [0, 3, 0],
        [0, -3, 0],
        [0, 1, 0]
    ], dtype=np.float32),
    np.array([
        [0, 0, 3],
        [0, -3, 0],
        [1, 0, 0]
    ], dtype=np.float32),
    np.array([
        [0, 0, 0],
        [1, -3, 3],
        [0, 0, 0]
    ], dtype=np.float32),
    np.array([
        [1, 0, 0],
        [0, -3, 0],
        [0, 0, 3]
    ], dtype=np.float32),
    np.array([
        [0, 1, 0],
        [0, -3, 0],
        [0, 3, 0]
    ], dtype=np.float32),
    np.array([
        [0, 0, 1],
        [0, -3, 0],
        [3, 0, 0]
    ], dtype=np.float32),
    np.array([
        [0, 0, 0],
        [3, -3, 1],
        [0, 0, 0],
    ], dtype=np.float32)
]

filter_edge_3x3 = [
    np.array([
        [-1, 2, -1],
        [2, -4, 2],
        [0, 0, 0]
    ], dtype=np.float32),
    np.array([
        [0, 2, -1],
        [0, -4, 2],
        [0, 2, -1]
    ], dtype=np.float32),
    np.array([
        [0, 0, 0],
        [2, -4, 2],
        [-1, 2, -1]
    ], dtype=np.float32),
    np.array([
        [-1, 2, 0],
        [2, -4, 0],
        [-1, 2, 0]
    ], dtype=np.float32),
]

filter_edge_5x5 = [
    np.array([
        [-1, 2, -2, 2, -1],
        [2, -6, 8, -6, 2],
        [-2, 8, -12, 8, -2],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ], dtype=np.float32),
    np.array([
        [0, 0, -2, 2, -1],
        [0, 0, 8, -6, 2],
        [0, 0, -12, 8, -2],
        [0, 0, 8, -6, 2],
        [0, 0, -2, 2, -1]
    ], dtype=np.float32),
    np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [-2, 8, -12, 8, -2],
        [2, -6, 8, -6, 2],
        [-1, 2, -2, 2, -1]
    ], dtype=np.float32),
    np.array([
        [-1, 2, -2, 0, 0],
        [2, -6, 8, 0, 0],
        [-2, 8, -12, 0, 0],
        [2, -6, 8, 0, 0],
        [-1, 2, -2, 0, 0]
    ], dtype=np.float32),
]

square_3x3 = np.array([
    [-1, 2, -1],
    [2, -4, 2],
    [-1, 2, -1]
], dtype=np.float32)

square_5x5 = np.array([
    [-1, 2, -2, 2, -1],
    [2, -6, 8, -6, 2],
    [-2, 8, -12, 8, -2],
    [2, -6, 8, -6, 2],
    [-1, 2, -2, 2, -1]
], dtype=np.float32)

all_hpf_list = filter_class_1 + filter_class_2 + filter_class_3 + filter_edge_3x3 + filter_edge_5x5 + [square_3x3,
                                                                                                       square_5x5]

hpf_3x3_list = filter_class_1 + filter_class_2 + filter_class_3 + filter_edge_3x3 + [square_3x3]
hpf_5x5_list = filter_edge_5x5 + [square_5x5]

normalized_filter_class_2 = [hpf / 2 for hpf in filter_class_2]
normalized_filter_class_3 = [hpf / 3 for hpf in filter_class_3]
normalized_filter_edge_3x3 = [hpf / 4 for hpf in filter_edge_3x3]
normalized_square_3x3 = square_3x3 / 4
normalized_filter_edge_5x5 = [hpf / 12 for hpf in filter_edge_5x5]
normalized_square_5x5 = square_5x5 / 12

all_normalized_hpf_list = filter_class_1 + normalized_filter_class_2 + normalized_filter_class_3 + \
                          normalized_filter_edge_3x3 + normalized_filter_edge_5x5 + [normalized_square_3x3,
                                                                                     normalized_square_5x5]

normalized_hpf_3x3_list = filter_class_1 + normalized_filter_class_2 + normalized_filter_class_3 + \
                          normalized_filter_edge_3x3 + [normalized_square_3x3]
normalized_hpf_5x5_list = normalized_filter_edge_5x5 + [normalized_square_5x5]


# Image preprocessing
# High-pass filters (HPF)
class HPF(nn.Module):
    def __init__(self):
        super(HPF, self).__init__()

        weight_3x3 = nn.Parameter(torch.Tensor(normalized_hpf_3x3_list).view(25, 1, 3, 3), requires_grad=False)
        weight_5x5 = nn.Parameter(torch.Tensor(normalized_hpf_5x5_list).view(5, 1, 5, 5), requires_grad=False)

        self.preprocess_3x3 = nn.Conv2d(1, 25, kernel_size=(3, 3), bias=False)
        with torch.no_grad():
            self.preprocess_3x3.weight = weight_3x3

        self.preprocess_5x5 = nn.Conv2d(1, 30, kernel_size=(5, 5), padding=(1, 1), bias=False)
        with torch.no_grad():
            self.preprocess_5x5.weight = weight_5x5

    def forward(self, x):
        processed3x3 = self.preprocess_3x3(x)
        processed5x5 = self.preprocess_5x5(x)

        # concatenate two tensors
        #   in:  torch.Size([2, 1,256,256])
        #   out: torch.Size([2, 30, 254, 254])
        output = torch.cat((processed3x3, processed5x5), dim=1)
        output = nn.functional.relu(output)

        return output


# Absolut value activation (ABS)
class ABS(nn.Module):
    def __init__(self):
        super(ABS, self).__init__()

    def forward(self, x):
        output = torch.abs(x)
        return output


class ADNetV3(nn.Module):
    def __init__(self):
        super(ADNetV3, self).__init__()

        # <------    Preprocessing module    ------>
        self.preprocess = HPF()

        # <------    Convolution module    ------>
        self.separable_convolution_1 = nn.Sequential(
            nn.Conv2d(30, 60, kernel_size=(3, 3), padding=(1, 1), groups=30),
            ABS(),
            nn.BatchNorm2d(60),
            nn.Conv2d(60, 30, kernel_size=(1, 1)),
            nn.BatchNorm2d(30),
            nn.ReLU()
        )
        self.separable_convolution_2 = nn.Sequential(
            nn.Conv2d(30, 60, kernel_size=(3, 3), padding=(1, 1), groups=30),
            nn.BatchNorm2d(60),
            nn.Conv2d(60, 30, kernel_size=(1, 1)),
            nn.BatchNorm2d(30),
            nn.ReLU()
        )
        self.base_block_1 = nn.Sequential(
            nn.Conv2d(30, 30, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(30),
            nn.ReLU(),
            nn.AvgPool2d(3, stride=2, padding=1)
        )
        self.base_block_2 = nn.Sequential(
            nn.Conv2d(30, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(3, stride=2, padding=1)
        )
        self.base_block_3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AvgPool2d(3, stride=2, padding=1)
        )
        self.base_block_4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # <------    Classification module   ------>

        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 1024)
        self.fc3 = nn.Linear(1024, 2)

    def forward(self, x):
        output = self.preprocess(x)

        output = self.separable_convolution_1(output)
        output = self.separable_convolution_2(output)
        output = self.base_block_1(output)
        output = self.base_block_2(output)
        output = self.base_block_3(output)
        output = self.base_block_4(output)
        # output = F.adaptive_avg_pool2d(output, (1,1))
        output = output.view(-1, 128)

        output = F.relu(self.fc1(output))
        output = F.relu(self.fc2(output))
        output = self.fc3(output)

        return output

    def info(self, input_size):
        print(self)
        print(summary(self, input_size))
