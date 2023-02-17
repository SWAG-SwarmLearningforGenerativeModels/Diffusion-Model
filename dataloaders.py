import os
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from PIL import Image


def get_data(config):

    if config.data.dataset == 'landscape':

        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(80),
            torchvision.transforms.RandomResizedCrop(
                config.data.image_size, scale=(0.8, 1.0)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        dataset = torchvision.datasets.ImageFolder(
            config.data.train_path, transform=transforms)

        print(f'Length of training dataset: {len(dataset)}')

        dataloader = DataLoader(
            dataset, batch_size=config.training.batch_size, num_workers=config.data.num_workers, shuffle=True)
        return dataloader

    elif config.data.dataset == 'cifar10':

        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        trainset = torchvision.datasets.CIFAR10(root=config.data.train_path, train=True,
                                                download=True, transform=transform)

        testset = torchvision.datasets.CIFAR10(root=config.data.train_path, train=False,
                                               download=True, transform=transform)

        c_dataset = torch.utils.data.ConcatDataset(
            [trainset, testset])

        print(f'Length of training dataset: {len(c_dataset)}')

        trainloader = torch.utils.data.DataLoader(c_dataset, batch_size=config.training.batch_size,
                                                  shuffle=True, num_workers=config.data.num_workers)

        # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        return trainloader

    elif config.data.dataset == 'brain':

        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Grayscale(1),
            torchvision.transforms.Resize(
                (config.data.image_size, config.data.image_size)),
            torchvision.transforms.Normalize((0.5, ), (0.5, ))
        ])
        train_dataset = torchvision.datasets.ImageFolder(
            config.data.train_path, transform=transforms)

        test_dataset = torchvision.datasets.ImageFolder(
            config.data.test_path, transform=transforms)

        c_dataset = torch.utils.data.ConcatDataset(
            [train_dataset, test_dataset])
        print(f'Length of training dataset: {len(c_dataset)}')
        dataloader = DataLoader(
            c_dataset, batch_size=config.training.batch_size, num_workers=config.data.num_workers, shuffle=True)

        return dataloader

    elif config.data.dataset == 'xray':

        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Grayscale(1),
            torchvision.transforms.Resize(
                (config.data.image_size, config.data.image_size)),
            torchvision.transforms.Normalize((0.5, ), (0.5, ))
        ])
        train_dataset = torchvision.datasets.ImageFolder(
            config.data.train_path, transform=transforms)

        print(f'Length of training dataset: {len(train_dataset)}')
        dataloader = DataLoader(
            train_dataset, batch_size=config.training.batch_size, num_workers=config.data.num_workers, shuffle=True)

        return dataloader


def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)


def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()
