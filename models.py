# -*- coding: UTF-8 -*-

import torch
from torch import nn

class Net_MNIST(nn.Module):

    def __init__(self):
        super(Net_MNIST, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.AvgPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.AvgPool2d(kernel_size=2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=4 * 4 * 16, out_features=120),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(in_features=84, out_features=10)
        )

    def forward(self, input):
        conv1_output = self.conv1(input)
        conv2_output = self.conv2(conv1_output)
        conv2_output = conv2_output.reshape((-1, 4 * 4 * 16))
        fc1_output = self.fc1(conv2_output)
        fc2_output = self.fc2(fc1_output)
        fc3_output = self.fc3(fc2_output)
        
        return fc3_output, fc2_output
    
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsampling=False, expansion=1):
        super(BasicBlock, self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)

        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels*self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_features=out_channels*self.expansion)
            )
    
    def forward(self, x):
        if self.downsampling:
            residual = self.downsample(x)
        else:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsampling=False, expansion=4):
        super(Bottleneck, self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)

        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)

        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels*self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=out_channels*self.expansion)

        self.relu = nn.ReLU()

        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels*self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_features=out_channels*self.expansion)
            )
    
    def forward(self, x):
        if self.downsampling:
            residual = self.downsample(x)
        else:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out

class ResNet_CIFAR10(nn.Module):
    def __init__(self, block_type=BasicBlock, block_nums=[2, 2, 2, 2], num_classes=10, expansion=1):
        super(ResNet_CIFAR10, self).__init__()
        self.expansion = expansion
        self.in_channels = 64

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=64, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU()
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block_type, out_channels=64, block=block_nums[0], stride=1)
        self.layer2 = self._make_layer(block_type, out_channels=128, block=block_nums[1], stride=2)
        self.layer3 = self._make_layer(block_type, out_channels=256, block=block_nums[2], stride=2)
        self.layer4 = self._make_layer(block_type, out_channels=512, block=block_nums[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_features=512 * self.expansion, out_features=num_classes, bias=True)

    def _make_layer(self, block_type, out_channels, block, stride):
        layers = []
        downsampling = False

        if stride != 1 or self.in_channels != out_channels * self.expansion:
            downsampling = True

        layers.append(block_type(self.in_channels, out_channels, stride, downsampling=downsampling))
        self.in_channels = out_channels * self.expansion

        for _ in range(1, block):
            layers.append(block_type(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x_feature = x.view(x.size(0), -1)
        x_output = self.fc(x_feature)

        return x_output, x_feature

class ResNet_CIFAR100(nn.Module):
    def __init__(self, block_type=Bottleneck, block_nums=[3, 4, 6, 3], num_classes=100, expansion=4):
        super(ResNet_CIFAR100, self).__init__()
        self.expansion = expansion
        self.in_channels = 64

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=64, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU()
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block_type, out_channels=64, block=block_nums[0], stride=1)
        self.layer2 = self._make_layer(block_type, out_channels=128, block=block_nums[1], stride=2)
        self.layer3 = self._make_layer(block_type, out_channels=256, block=block_nums[2], stride=2)
        self.layer4 = self._make_layer(block_type, out_channels=512, block=block_nums[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_features=512 * self.expansion, out_features=num_classes, bias=True)

    def _make_layer(self, block_type, out_channels, block, stride):
        layers = []
        downsampling = False

        if stride != 1 or self.in_channels != out_channels * self.expansion:
            downsampling = True

        layers.append(block_type(self.in_channels, out_channels, stride, downsampling=downsampling))
        self.in_channels = out_channels * self.expansion

        for _ in range(1, block):
            layers.append(block_type(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x_feature = x.view(x.size(0), -1)
        x_output = self.fc(x_feature)

        return x_output, x_feature
    
def model_loader(model_name, pretrained_model_path=None):
    if model_name == "Net_MNIST":
        model = Net_MNIST()
    elif model_name == "ResNet_10":
        model = ResNet_CIFAR10()
    elif model_name == "ResNet_100":
        model = ResNet_CIFAR100()
    model.load_state_dict(torch.load(pretrained_model_path))

    return model
