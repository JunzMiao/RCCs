# -*- coding: utf-8 -*-

import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class Data_Loader():
    """
    Description
    -----------
        load dataset.

    Example
    -------
        >>> train_loader, test_loader = Data_Loader("mnist", "./data/", 64).load_data()
    """

    def __init__(self, dataset_name, data_path, train_batch_size, test_batch_size=1000):
        """
        Parameters
        ----------
            dataset_name : str, options={"mnist", "fashionmnist", "cifar10", "cifar100"}
            
            data_path : str

            train_batch_size : int

            test_batch_size : int, default=1000
        """

        self.dataset_name = dataset_name
        self.data_path = data_path
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
    
    def load_data(self):
        if self.dataset_name == 'mnist':
            return load_mnist(self.data_path, self.train_batch_size, self.test_batch_size)
        elif self.dataset_name == 'fashionmnist':
            return load_fashionmnist(self.data_path, self.train_batch_size, self.test_batch_size)
        elif self.dataset_name == 'cifar10':
            return load_cifar10(self.data_path, self.train_batch_size, self.test_batch_size)
        elif self.dataset_name == 'cifar100':
            return load_cifar100(self.data_path, self.train_batch_size, self.test_batch_size)


def load_mnist(data_path, train_batch_size, test_batch_size):
    train_data = datasets.MNIST(
        root=data_path,
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor()
        ])
    )
    test_data = datasets.MNIST(
        root=data_path,
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor()
        ])
    )

    train_dataloader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False)

    return train_dataloader, test_dataloader


def load_fashionmnist(data_path, train_batch_size, test_batch_size):
    train_data = datasets.FashionMNIST(
        root=data_path,
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor()
        ])
    )
    test_data = datasets.FashionMNIST(
        root=data_path,
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor()
        ])
    )

    train_dataloader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False)

    return train_dataloader, test_dataloader


def load_cifar10(data_path, train_batch_size, test_batch_size):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_data = datasets.CIFAR10(
        root=data_path,
        train=True,
        download=True,
        transform=transform_train,
    )
    test_data = datasets.CIFAR10(
        root=data_path,
        train=False,
        download=True,
        transform=transform_test
    )

    train_dataloader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False)

    return train_dataloader, test_dataloader


def load_cifar100(data_path, train_batch_size, test_batch_size):
    train_data = datasets.CIFAR100(
        root=data_path,
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[5879, 0.2564, 0.2762])
        ])
    )
    test_data = datasets.CIFAR100(
        root=data_path,
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[5879, 0.2564, 0.2762])
        ])
    )

    train_dataloader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False)

    return train_dataloader, test_dataloader
