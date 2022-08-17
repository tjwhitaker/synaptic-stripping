import torch
import torchvision
import numpy as np
import os
from autoaugment import CIFAR10Policy, SVHNPolicy


def get_loaders(dataset='cifar10', data_path='data', autoaugment=True, batch_size=128, num_workers=4):
    if dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2471, 0.2435, 0.2616)

        train_transforms = [
            torchvision.transforms.RandomCrop(32, padding=4, fill=128),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean, std)]

        test_transforms = [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean, std)]

        if autoaugment:
            train_transforms.insert(2, CIFAR10Policy())

        train_transforms = torchvision.transforms.Compose(train_transforms)
        test_transforms = torchvision.transforms.Compose(test_transforms)

        train_dataset = torchvision.datasets.CIFAR10(
            data_path, train=True, transform=train_transforms, download=True)

        test_dataset = torchvision.datasets.CIFAR10(
            data_path, train=False, transform=test_transforms, download=True)

    elif dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)

        train_transforms = [
            torchvision.transforms.RandomCrop(32, padding=4, fill=128),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean, std)]

        test_transforms = [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean, std)]

        if autoaugment:
            train_transforms.insert(2, CIFAR10Policy())

        train_transforms = torchvision.transforms.Compose(train_transforms)
        test_transforms = torchvision.transforms.Compose(test_transforms)

        train_dataset = torchvision.datasets.CIFAR100(
            data_path, train=True, transform=train_transforms, download=True)

        test_dataset = torchvision.datasets.CIFAR100(
            data_path, train=False, transform=test_transforms, download=True)

    elif dataset == 'svhn':
        mean = (0.4377, 0.4438, 0.4728)
        std = (0.1980, 0.2010, 0.1970)

        train_transforms = [
            torchvision.transforms.RandomCrop(32, padding=4, fill=128),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean, std)]

        test_transforms = [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean, std)]

        if autoaugment:
            train_transforms.insert(1, SVHNPolicy())

        train_transforms = torchvision.transforms.Compose(train_transforms)
        test_transforms = torchvision.transforms.Compose(test_transforms)

        train_dataset = torchvision.datasets.SVHN(
            data_path, split="train", transform=train_transforms, download=True)

        test_dataset = torchvision.datasets.SVHN(
            data_path, split="test", transform=test_transforms, download=True)

    elif dataset == 'tinyimagenet':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        train_transforms = [
            torchvision.transforms.Resize(32),
            torchvision.transforms.RandomCrop(32, padding=4, fill=128),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean, std)]

        test_transforms = [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean, std)]

        if autoaugment:
            train_transforms.insert(3, CIFAR10Policy())

        train_transforms = torchvision.transforms.Compose(train_transforms)
        test_transforms = torchvision.transforms.Compose(test_transforms)

        # Need to download tiny imagenet manually
        train_dataset = torchvision.datasets.ImageFolder(
            f"{data_path}/tiny-imagenet-200/train", transform=train_transforms)

        test_dataset = torchvision.datasets.ImageFolder(
            f"{data_path}/tiny-imagenet-200/val", transform=test_transforms)

    else:
        print(f"Dataset <{dataset}> is not supported")
        raise

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False)

    return train_loader, test_loader


def get_corrupted_loader(dataset='cifar10', data_path='data', batch_size=128, num_workers=4):
    corruptions = [
        "brightness",
        "contrast",
        "defocus_blur",
        "elastic_transform",
        "fog",
        "frost",
        "gaussian_blur",
        "gaussian_noise",
        "glass_blur",
        "impulse_noise",
        "jpeg_compression",
        "motion_blur",
        "pixelate",
        "saturate",
        "shot_noise",
        "snow",
        "spatter",
        "speckle_noise",
        "zoom_blur"
    ]

    if dataset == "cifar10":
        num_classes = 10
        labels_path = f"{data_path}/CIFAR-10-C/labels.npy"

        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2471, 0.2435, 0.2616)

        images = []

        for c in corruptions:
            data = torch.from_numpy(
                np.load(f"{data_path}/CIFAR-10-C/{c}.npy")).permute(0, 3, 1, 2) / 255.0
            images.append(data)

        images = torch.cat(images, dim=0)

        for image in images:
            image[0] = (image[0] - mean[0]) / std[0]
            image[1] = (image[1] - mean[1]) / std[1]
            image[2] = (image[2] - mean[2]) / std[2]

        labels = np.load(labels_path)
        labels = np.repeat([labels], len(corruptions), axis=0).flatten()

        dataset = torch.utils.data.TensorDataset(
            images, torch.from_numpy(labels))

        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False)

    elif dataset == "cifar100":
        num_classes = 100
        labels_path = f"{data_path}/CIFAR-100-C/labels.npy"

        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)

        images = []

        for c in corruptions:
            data = torch.from_numpy(
                np.load(f"{data_path}/CIFAR-100-C/{c}.npy")).permute(0, 3, 1, 2) / 255.0
            images.append(data)

        images = torch.cat(images, dim=0)

        for image in images:
            image[0] = (image[0] - mean[0]) / std[0]
            image[1] = (image[1] - mean[1]) / std[1]
            image[2] = (image[2] - mean[2]) / std[2]

        labels = np.load(labels_path)
        labels = np.repeat([labels], len(corruptions), axis=0).flatten()

        dataset = torch.utils.data.TensorDataset(
            images, torch.from_numpy(labels))

        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False)

    elif dataset == "svhn":
        num_classes = 10
        labels_path = f"{data_path}/SVHN-C/labels.npy"

        mean = (0.4377, 0.4438, 0.4728)
        std = (0.1980, 0.2010, 0.1970)

        images = torch.zeros((2473040, 3, 32, 32))

        for i in range(len(corruptions)):
            c = corruptions[i]
            data = torch.from_numpy(
                np.load(f"{data_path}/SVHN-C/{c}.npy")).permute(0, 3, 1, 2) / 255.0
            start_index = i*130160
            end_index = start_index + 130160
            images[start_index:end_index, :, :, :] = data

        # images = torch.cat(images, dim=0)
        print("images concatenated")

        for image in images:
            image[0] = (image[0] - mean[0]) / std[0]
            image[1] = (image[1] - mean[1]) / std[1]
            image[2] = (image[2] - mean[2]) / std[2]

        labels = np.load(labels_path)
        labels = np.repeat([labels], len(corruptions), axis=0).flatten()

        dataset = torch.utils.data.TensorDataset(
            images, torch.from_numpy(labels))

        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False)

    else:
        num_classes = 200
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        transforms = torchvision.transforms.Compose(
            [torchvision.transforms.Resize(32),
             torchvision.transforms.ToTensor(),
             torchvision.transforms.Normalize(mean, std)])

        datasets = []

        for c in os.listdir(f"{data_path}/Tiny-ImageNet-C"):
            for s in range(1, 6):
                datasets.append(torchvision.datasets.ImageFolder(
                    f"{data_path}/Tiny-ImageNet-C/{c}/{s}", transform=transforms))

        loader = torch.utils.data.DataLoader(
            torch.utils.data.ConcatDataset(datasets),
            batch_size=batch_size,
            num_workers=num_workers)

    return loader, num_classes
