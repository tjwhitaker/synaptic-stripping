import torch
import torchvision
from autoaugment import CIFAR10Policy, SVHNPolicy


def get_loaders(dataset='cifar10', data_path='data', autoaugment=True, batch_size=128, num_workers=4):
    if dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2471, 0.2435, 0.2616)

        train_transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4, fill=128),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean, std)])

        test_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean, std)])

        train_dataset = torchvision.datasets.CIFAR10(
            "data", train=True, transform=train_transforms, download=True)

        test_dataset = torchvision.datasets.CIFAR10(
            "data", train=False, transform=test_transforms, download=True)

    elif dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)

        train_transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4, fill=128),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean, std)])

        test_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean, std)])

        train_dataset = torchvision.datasets.CIFAR100(
            "data", train=True, transform=train_transforms, download=True)

        test_dataset = torchvision.datasets.CIFAR100(
            "data", train=False, transform=test_transforms, download=True)

    elif dataset == 'svhn':
        mean = (0.4377, 0.4438, 0.4728)
        std = (0.1980, 0.2010, 0.1970)

        train_transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4, fill=128),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean, std)])

        test_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean, std)])

        train_dataset = torchvision.datasets.SVHN(
            "data", split="train", transform=train_transforms, download=True)

        test_dataset = torchvision.datasets.SVHN(
            "data", split="test", transform=test_transforms, download=True)

    elif dataset == 'tinyimagenet':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        train_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(32),
            torchvision.transforms.RandomCrop(32, padding=4, fill=128),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean, std)])

        test_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean, std)])

        # Need to download tiny imagenet manually
        train_dataset = torchvision.datasets.ImageFolder(
            "data/tiny-imagenet-200/train", transform=train_transforms)

        test_dataset = torchvision.datasets.ImageFolder(
            "data/tiny-imagenet-200/val", transform=test_transforms)

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
