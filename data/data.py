# data.py

import os
import PIL
import numpy as np
import h5py
import tarfile
import os
import shutil

import torch
import pytorch_lightning as pl

from typing import Tuple, Union, List
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader, ConcatDataset, TensorDataset, random_split
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split, Dataset
from medmnist import PathMNIST

from tools.utils import RandomResizedCrop

class MedMNISTWrapper(Dataset):
    '''
    Wraps MedMNIST dataset to remove singleton dimensions from labels.
    '''

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        y = y.squeeze()
        return x, y

def _preprocess_tiny_imagenet_train(train_dir: str):
    '''
    Preprocess Tiny ImageNet train set.
    '''

    for class_name in os.listdir(train_dir):
        class_path = os.path.join(train_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        images_dir = os.path.join(class_path, "images")
        if os.path.exists(images_dir):
            for fname in os.listdir(images_dir):
                if fname.lower().endswith((".jpeg", ".jpg", ".png")):
                    shutil.move(os.path.join(images_dir, fname), class_path)
            shutil.rmtree(images_dir)

        # remove any .txt files inside the class folder
        for fname in os.listdir(class_path):
            if fname.endswith(".txt"):
                os.remove(os.path.join(class_path, fname))

def _preprocess_tiny_imagenet_val(val_dir: str):
    '''
    Preprocess Tiny ImageNet val set.
    '''
    
    print(f"Looking for annotations in: {os.path.join(val_dir, 'val_annotations.txt')}")

    val_img_dir = os.path.join(val_dir, "images")
    ann_file = os.path.join(val_dir, "val_annotations.txt")

    if not os.path.exists(ann_file):
        print("val_annotations.txt not found, skipping.")
        return

    with open(ann_file, "r") as f:
        annotations = [line.strip().split("\t") for line in f]

    for img_file, cls, *_ in annotations:
        cls_dir = os.path.join(val_dir, cls)
        os.makedirs(cls_dir, exist_ok=True)
        src_path = os.path.join(val_img_dir, img_file)
        dst_path = os.path.join(cls_dir, img_file)

        if os.path.exists(src_path):
            shutil.move(src_path, dst_path)

    if os.path.exists(val_img_dir):
        shutil.rmtree(val_img_dir)

def load_tiny_imagenet(data_path: str, batch_size: int, num_workers: int, use_validation: bool,
                       train_transform, eval_transform) -> Tuple[DataLoader, Union[DataLoader, None], DataLoader]:
    '''
    Loads Tiny ImageNet, applies preprocessing, and returns data loaders.
    '''

    dataset_root = os.path.join(data_path, 'tiny-imagenet-200')
    if not os.path.exists(dataset_root):
        raise FileNotFoundError(f"Tiny ImageNet not found at {dataset_root}. Please download and extract it.")

    train_dir = os.path.join(dataset_root, 'train')
    val_dir = os.path.join(dataset_root, 'val')

    # preprocess both splits
    _preprocess_tiny_imagenet_train(train_dir)
    _preprocess_tiny_imagenet_val(val_dir)

    train_dataset = ImageFolder(root=train_dir, transform=train_transform)
    test_dataset = ImageFolder(root=val_dir, transform=eval_transform)

    if use_validation:

        train_size = int(0.9 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size],
                                                  generator=torch.Generator().manual_seed(0))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  drop_last=True, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                drop_last=True, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                 drop_last=True, num_workers=num_workers)
        
        return train_loader, val_loader, test_loader
    
    else:

        full_dataset = ConcatDataset([train_dataset, test_dataset])
        train_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True,
                                  drop_last=True, num_workers=num_workers)
        return train_loader, None, None

def load_flowers(data_path: str, batch_size: int, num_workers: int, use_validation: bool, 
                   train_transform, eval_transform) -> Tuple[DataLoader, Union[DataLoader, None], DataLoader]:
    '''
    Preprocess Flowers17 dataset.
    '''
    tar_path = os.path.join(data_path, '17flowers.tgz')
    extracted_path = os.path.join(data_path, 'jpg')

    # extract if not already extracted
    if not os.path.exists(extracted_path):
        if os.path.exists(tar_path):
            print(f"Extracting {tar_path}...")
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(path=data_path)
            print("Extraction complete.")
        else:
            raise FileNotFoundError(f"{tar_path} not found. Please make sure the file exists.")

    dataset = ImageFolder(root=extracted_path, transform=train_transform)

    if use_validation:
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(0))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(ImageFolder(root=extracted_path, transform=eval_transform), batch_size=batch_size, shuffle=False, num_workers=num_workers)

        return train_loader, val_loader, test_loader
    else:
        combined_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        return combined_loader, [], []

def organize_flowers(data_dir):
    '''
    Restructure flat image dir into class-specific subdirs.
    '''

    class_names = ['Daffodil', 'Snowdrop', 'Lily Valley', 'Bluebell',
                   'Crocus', 'Iris', 'Tigerlily', 'Tulip',
                   'Fritillary', 'Sunflower', 'Daisy', 'Colts Foot',
                   'Dandelion', 'Cowslip', 'Buttercup', 'Windflower', 'Pansy']
    images = sorted([f for f in os.listdir(data_dir) if f.endswith('.jpg')])
    
    num_images_per_class = 80  # 80 images per class

    for idx, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        for i in range(num_images_per_class):
            img_idx = idx * num_images_per_class + i
            if img_idx < len(images):
                src_path = os.path.join(data_dir, images[img_idx])
                dst_path = os.path.join(class_dir, images[img_idx])
                shutil.move(src_path, dst_path)
            else:
                break

def load_usps(data_path: str = './data/datasets', batch_size: int = 256, num_workers: int = 0, use_validation: bool = False) -> Tuple[DataLoader, Union[DataLoader, None], DataLoader]:
    '''
    Load USPS dataset from HDF5 file.
    '''

    with h5py.File(os.path.join(data_path, 'usps.h5'), 'r') as hf:
        train = hf.get('train')
        x_train = train.get('data')[:]
        y_train = train.get('target')[:]

        test = hf.get('test')
        x_test = test.get('data')[:]
        y_test = test.get('target')[:]

    x_train_tensor = torch.tensor(x_train.reshape(-1, 16, 16), dtype=torch.float32).unsqueeze(1)
    x_test_tensor = torch.tensor(x_test.reshape(-1, 16, 16), dtype=torch.float32).unsqueeze(1)

    y_train, y_test = map(lambda y: torch.tensor(y).long(), (y_train, y_test))

    train_dataset = TensorDataset(x_train_tensor, y_train)
    test_dataset = TensorDataset(x_test_tensor, y_test)

    if use_validation:
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size

        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(0))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        return train_loader, val_loader, test_loader
    else:
        combined_loader = DataLoader(TensorDataset(torch.cat((x_train_tensor, x_test_tensor)), torch.cat((y_train, y_test))), batch_size=batch_size, shuffle=True, num_workers=num_workers)
        return combined_loader, [], []

def load_reuters(data_path: str = './data/datasets', batch_size: int = 256, num_workers: int = 0, use_validation: bool = False) -> Tuple[DataLoader, Union[DataLoader, None], DataLoader]:
    '''
    Load Reuters dataset from numpy file.
    '''

    data = np.load(os.path.join(data_path, 'reutersidf10k.npy'), allow_pickle=True).item()
    x = torch.tensor(data['data'], dtype=torch.float32)
    y = torch.tensor(data['label'], dtype=torch.long)

    dataset = TensorDataset(x, y)
    if use_validation:
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=num_workers)
        return train_loader, val_loader, []
    else:
        combined_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
        return combined_loader, [], []

def build_transform(is_train: bool, 
                    dataset_name: str, 
                    input_size: int, 
                    num_channels: int,
                    horizontal_flip: float = 0.5, 
                    randaug_n: int = 2, 
                    resize_scale: tuple = (0.08, 1.0), 
                    resize_ratio: tuple = (0.75, 1.3333), 
                    reprob: float = 0.25, 
                    remode: str = 'pixel', 
                    recount: int = 1, 
                    aa: str = 'rand-m9-mstd0.5-inc1'):
    '''
    Constructs image transformation pipelines for training/eval.
    '''

    if dataset_name in ['mnist', 'fmnist', 'usps']:
        return transforms.Compose([
            transforms.ToTensor(),
        ])
    else:
        if num_channels == 1:
            mean = (0.5,)
            std = (0.5,)
        elif dataset_name in ['cifar-10', 'cifar-100']:
            mean = (0.4914, 0.4822, 0.4465)
            std = (0.2023, 0.1994, 0.2010)
        elif dataset_name in ['medmnist']:
            mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        else:
            mean = IMAGENET_DEFAULT_MEAN
            std = IMAGENET_DEFAULT_STD

        if is_train:
            timm_transform = create_transform(
                input_size=input_size,
                is_training=True,
                auto_augment=aa,
                interpolation='bicubic',
                re_prob=reprob,
                re_mode=remode,
                re_count=recount,
                mean=mean,
                std=std,
            )
            transform = transforms.Compose([
                RandomResizedCrop(input_size, scale=resize_scale, ratio=resize_ratio, interpolation=InterpolationMode.BICUBIC),
                transforms.RandAugment(num_ops=randaug_n),
                transforms.RandomHorizontalFlip(p=horizontal_flip),
                timm_transform
            ])
        else:
            t = []
            crop_pct = 1.0 if dataset_name in ['mnist', 'fmnist', 'usps'] else 0.875 if input_size <= 224 else 1.0  # Set crop_pct=1.0 for MNIST to prevent resizing
            size = int(input_size / crop_pct)
            t.append(transforms.Resize(size, interpolation=PIL.Image.BICUBIC))
            t.append(transforms.CenterCrop(input_size))
            t.append(transforms.ToTensor())
            t.append(transforms.Normalize(mean, std))
            transform = transforms.Compose(t)

        return transform

def get_dataloaders(dataset_name: str, 
                    batch_size: int, 
                    num_workers: int, 
                    use_validation: bool = False, 
                    horizontal_flip: float = 0.5, 
                    randaug_n: int = 2, 
                    resize_scale: tuple = (0.08, 1.0), 
                    resize_ratio: tuple = (0.75, 1.3333), 
                    reprob: float = 0.25,
                    remode: str = 'pixel',
                    recount: int = 1.0,
                    autoaugment: bool = False, 
                    input_size: int = 224,
                    num_channels: int = 3) -> Tuple[DataLoader, Union[DataLoader, None], DataLoader]:
    '''
    Create dataloaders for a given dataset.
    '''
    
    transform = build_transform(is_train=True, dataset_name=dataset_name, input_size=input_size, num_channels=num_channels, 
                                horizontal_flip=horizontal_flip, randaug_n=randaug_n, resize_scale=resize_scale, resize_ratio=resize_ratio, 
                                reprob=reprob, remode=remode, recount=recount, aa='rand-m9-mstd0.5-inc1' if autoaugment else None)
    
    eval_transform = build_transform(is_train=False, dataset_name=dataset_name, input_size=input_size, num_channels=num_channels)

    if dataset_name == 'usps':
        print(f'Creating loaders for USPS...')
        return load_usps(data_path='./data/datasets', batch_size=batch_size, num_workers=num_workers, use_validation=use_validation)
    elif dataset_name == 'reuters-10k':
        print(f'Creating loaders for Reuters 10k...')
        return load_reuters(data_path='./data/datasets', batch_size=batch_size, num_workers=num_workers, use_validation=use_validation)
    elif dataset_name == 'flowers-17':
        print(f'Creating loaders for Flowers 17...')
        organize_flowers('./data/datasets/jpg')
        return load_flowers(data_path='./data/datasets', batch_size=batch_size, num_workers=num_workers, 
                              use_validation=use_validation, train_transform=transform, eval_transform=eval_transform)
    elif dataset_name == 'tiny-imagenet':
        return load_tiny_imagenet(data_path='./data/datasets', batch_size=batch_size, num_workers=num_workers,
                                  use_validation=use_validation, train_transform=transform, eval_transform=eval_transform)

    dataset_class = {
        'mnist': datasets.MNIST,
        'fmnist': datasets.FashionMNIST,
        'cifar-10': datasets.CIFAR10,
        'cifar-100': datasets.CIFAR100,
        'flowers-102': datasets.Flowers102,
        'svhn': datasets.SVHN,
        'medmnist': PathMNIST
    }.get(dataset_name)

    if dataset_class is None:
        raise ValueError(f"Dataset {dataset_name} is not supported")

    print(f'Creating loaders for {dataset_name}')
    if dataset_name == 'medmnist':
        train_dataset = dataset_class(split='train', transform=transform, download=True, root='./data/datasets')
        test_dataset = dataset_class(split='test', transform=eval_transform, download=True, root='./data/datasets')
        train_dataset = MedMNISTWrapper(train_dataset)
        test_dataset = MedMNISTWrapper(test_dataset)
    elif dataset_name in ['mnist', 'fmnist', 'cifar-10', 'cifar-100']:
        train_dataset = dataset_class(root='./data/datasets', train=True, download=True, transform=transform)
        test_dataset = dataset_class(root='./data/datasets', train=False, download=True, transform=eval_transform)
    else:
        train_dataset = dataset_class(root='./data/datasets', split='train', download=True, transform=transform)
        test_dataset = dataset_class(root='./data/datasets', split='test', download=True, transform=eval_transform)

    if use_validation:
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(0))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=num_workers)

        return train_loader, val_loader, test_loader
    else:
        train_loader = DataLoader(ConcatDataset([train_dataset, test_dataset]), batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
        return train_loader, [], []
