"""
Create Dataloaders and initialize data augmentation

@author Alan Luo
@version 1.1
"""

import os
import shutil
import tarfile
import h5py
import PIL.Image
import torch

from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Type
from tools.utils import RandomResizedCrop, ConfigObject
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from timm.data import create_transform, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from torch.utils.data import (
    DataLoader, ConcatDataset, TensorDataset, 
    random_split, Dataset
)

# Constants
DEFAULT_DATA_PATH = "./data/datasets"
FLOWERS_CLASS_NAMES = [
    'Daffodil', 'Snowdrop', 'Lily Valley', 'Bluebell',
    'Crocus', 'Iris', 'Tigerlily', 'Tulip',
    'Fritillary', 'Sunflower', 'Daisy', 'Colts Foot',
    'Dandelion', 'Cowslip', 'Buttercup', 'Windflower', 'Pansy'
]
NUM_IMAGES_PER_CLASS = 80
SUPPORTED_DATASETS = [
    'mnist', 'fmnist', 'cifar-10', 'cifar-100',
    'flowers-102', 'svhn', 'usps', 'reuters-10k', 'flowers-17'
]

@dataclass
class DatasetConfig:
    """Enhanced dataset configuration with type hints and defaults"""
    dataset_name: str
    input_size: int
    num_channels: int
    horizontal_flip: float = 0.5
    randaug_n: int = 2
    resize_scale: tuple = (0.08, 1.0)
    resize_ratio: tuple = (0.75, 1.3333)
    reprob: float = 0.25
    remode: str = 'pixel'
    recount: int = 1
    autoaugment: bool = False

def organize_flowers(data_dir: str) -> None:
    """Organize Flowers-17 images into class-specific directories.
    
    Args:
        data_dir: Path containing raw JPG images
    """
    images = sorted(f for f in os.listdir(data_dir) if f.endswith('.jpg'))
    
    for idx, class_name in enumerate(FLOWERS_CLASS_NAMES):
        class_dir = os.path.join(data_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        for i in range(NUM_IMAGES_PER_CLASS):
            img_idx = idx * NUM_IMAGES_PER_CLASS + i
            if img_idx >= len(images):
                break
            src = os.path.join(data_dir, images[img_idx])
            dst = os.path.join(class_dir, images[img_idx])
            shutil.move(src, dst)

def _create_splits(
    dataset: Dataset,
    eval_dataset: Dataset,
    use_validation: bool,
    batch_size: int,
    num_workers: int
) -> Tuple[DataLoader, Optional[DataLoader], DataLoader]:
    """Create train/val/test splits across datasets."""
    if not use_validation:
        loader = DataLoader(
            ConcatDataset([dataset, eval_dataset]),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )
        return loader, None, None
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(0)
    )
    
    return (
        DataLoader(train_set, batch_size, True, num_workers=num_workers),
        DataLoader(val_set, batch_size, False, num_workers=num_workers),
        DataLoader(eval_dataset, batch_size, False, num_workers=num_workers)
    )

def load_flowers(
    data_path: str = DEFAULT_DATA_PATH,
    batch_size: int = 32,
    num_workers: int = 0,
    use_validation: bool = False,
    train_transform: Optional[transforms.Compose] = None,
    eval_transform: Optional[transforms.Compose] = None
) -> Tuple[DataLoader, Optional[DataLoader], DataLoader]:
    """Load Flowers-17 dataset from tar file.
    
    Args:
        data_path: Root directory for dataset files
        batch_size: Samples per batch
        num_workers: Parallel loading threads
        use_validation: Whether to create validation split
        train_transform: Transformations for training data
        eval_transform: Transformations for evaluation data
    """
    tar_path = os.path.join(data_path, '17flowers.tgz')
    extracted_path = os.path.join(data_path, 'jpg')

    if not os.path.exists(extracted_path):
        if not os.path.exists(tar_path):
            raise FileNotFoundError(f"Flowers dataset not found at {tar_path}")
        
        print(f"Extracting {tar_path}...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(data_path)
        
        if not os.path.exists(extracted_path):
            raise RuntimeError(f"Failed to extract {tar_path}")

    organize_flowers(extracted_path)
    train_set = datasets.ImageFolder(extracted_path, train_transform)
    test_set = datasets.ImageFolder(extracted_path, eval_transform)
    
    return _create_splits(
        train_set, test_set, use_validation, 
        batch_size, num_workers
    )

def load_usps(
    data_path: str = DEFAULT_DATA_PATH,
    batch_size: int = 256,
    num_workers: int = 0,
    use_validation: bool = False
) -> Tuple[DataLoader, Optional[DataLoader], DataLoader]:
    """Load USPS dataset from HDF5 file."""
    usps_path = os.path.join(data_path, 'usps.h5')
    
    with h5py.File(usps_path, 'r') as hf:
        train_data = hf['train'][:]
        test_data = hf['test'][:]
    
    x_train = torch.tensor(train_data['data'].reshape(-1, 16, 16), dtype=torch.float32).unsqueeze(1)
    y_train = torch.tensor(train_data['target'], dtype=torch.long)
    x_test = torch.tensor(test_data['data'].reshape(-1, 16, 16), dtype=torch.float32).unsqueeze(1)
    y_test = torch.tensor(test_data['target'], dtype=torch.long)

    return _create_splits(
        TensorDataset(x_train, y_train),
        TensorDataset(x_test, y_test),
        use_validation, batch_size, num_workers
    )

def build_transform(config: ConfigObject, is_train: bool) -> transforms.Compose:
    """Build transforms using direct config access"""
    dataset_name = config.data.dataset
    input_size = config.data.input_size
    num_channels = config.data.num_channels
    augment = config.data.augment
    
    if dataset_name in ['mnist', 'fmnist', 'usps']:
        return transforms.Compose([transforms.ToTensor()])

    # normalization parameters
    if num_channels == 1:
        mean, std = (0.5,), (0.5,)
    elif dataset_name in ['cifar-10', 'cifar-100']:
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    else:
        mean, std = IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

    if is_train:
        return transforms.Compose([
            RandomResizedCrop(
                input_size,
                scale=tuple(augment.resize_scale),
                ratio=tuple(augment.resize_ratio),
                interpolation=InterpolationMode.BICUBIC
            ),
            transforms.RandAugment(num_ops=augment.randaug_n),
            transforms.RandomHorizontalFlip(p=augment.horizontal_flip),
            create_transform(
                input_size=input_size,
                is_training=True,
                auto_augment='rand-m9-mstd0.5-inc1' if augment.autoaugment else None,
                interpolation='bicubic',
                re_prob=augment.reprob,
                re_mode=augment.remode,
                re_count=augment.recount,
                mean=mean,
                std=std,
            )
        ])
    
    # eval transforms
    crop_pct = 1.0 if dataset_name in ['mnist', 'fmnist', 'usps'] else 0.875
    size = int(input_size / crop_pct)
    
    return transforms.Compose([
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

def get_dataloaders(
    config: ConfigObject,
    batch_size: int,
    num_workers: int,
    use_validation: bool
) -> Tuple[DataLoader, Optional[DataLoader], DataLoader]:
    """Create standardized dataloaders for supported datasets.
    
    Args:
        dataset_name: Name of dataset to load
        batch_size: Samples per batch
        num_workers: Parallel loading threads
        use_validation: Whether to create validation split
        config: Transformation configuration parameters
    """
    if config.data.dataset not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Unsupported dataset '{config.data.dataset}'. "
            f"Supported: {SUPPORTED_DATASETS}"
        )
    
    dataset_name = config.data.dataset  # Set dynamically
    
    train_transform = build_transform(config, is_train=True)
    eval_transform = build_transform(config, is_train=False)

    if dataset_name == 'usps':
        return load_usps(DEFAULT_DATA_PATH, batch_size, num_workers, use_validation)
    if dataset_name == 'flowers-17':
        return load_flowers(
            DEFAULT_DATA_PATH, batch_size, num_workers,
            use_validation, train_transform, eval_transform
        )

    # Handle torchvision datasets
    dataset_map: Dict[str, Type[Dataset]] = {
        'mnist': datasets.MNIST,
        'fmnist': datasets.FashionMNIST,
        'cifar-10': datasets.CIFAR10,
        'cifar-100': datasets.CIFAR100,
        'flowers-102': datasets.Flowers102,
        'svhn': datasets.SVHN
    }
    
    ds_class = dataset_map[dataset_name]
    train_set = ds_class(DEFAULT_DATA_PATH, train=True, download=True, transform=train_transform)
    test_set = ds_class(DEFAULT_DATA_PATH, train=False, download=True, transform=eval_transform)
    
    return _create_splits(train_set, test_set, use_validation, batch_size, num_workers)