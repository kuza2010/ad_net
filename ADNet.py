# torch
import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary

# kernels
from srm_filter_kernel import normalized_hpf_5x5_list, normalized_hpf_3x3_list


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


class ADNet(nn.Module):
    def __init__(self):
        super(ADNet, self).__init__()

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
        self.fc1 = nn.Sequential(
            nn.Linear(128, 256),
            nn.Dropout2d()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(256, 1024),
            nn.Dropout2d()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(1024, 2),
            nn.Dropout2d()
        )

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

        output = self.fc1(output)
        output = self.fc2(output)
        output = self.fc3(output)

        return output

    def info(self, input_size):
        print(self)
        print(summary(self, input_size))
