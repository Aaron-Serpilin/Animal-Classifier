import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()
class_names = ''

def create_dataloaders(
        train_dir: str,
        test_dir: str,
        train_transform: transforms.Compose,
        test_transform: transforms.Compose,
        batch_size: int,
        num_workers: int=NUM_WORKERS
    ):

    """

    Creates the Train and Test DataLoaders.

    Given the Computer Vision nature of this project, we will use ImageFolder for the datasets,
    and the convert it into DataLoaders with their custom hyper parameters to iterate through the batches. 

    Args:
        train_dir: Path to the training directory
        test_dir: Path to the testing directory
        transform: Torchvision transforms to perform on the data
        batch_size: Number of samples in each batch
        num_workers: Number of subprocesses used to load the data

    Returns:
        A tuple of (train_dataloader, test_dataloader, class_names)
        
    """

    train_data = datasets.ImageFolder(train_dir, transform=train_transform)
    test_data = datasets.ImageFolder(test_dir, transform=test_transform)

    class_names = train_data.classes

    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True, # we shuffle here to avoid the model learning order from the data
        num_workers=num_workers,
        pin_memory=True
    )

    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_dataloader, test_dataloader, class_names