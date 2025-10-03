import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os
from src import config

def get_dataloaders():
    """
    Creates training and validation dataloaders.

    Returns:
        tuple: A tuple containing the training dataloader and validation dataloader.
    """
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(config.IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(config.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load the full dataset from the directory
    full_dataset = datasets.ImageFolder(root=config.DATA_DIR)

    # Split the dataset into training and validation sets
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    
    generator = torch.Generator().manual_seed(config.RANDOM_SEED)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform

    # Create DataLoaders for both sets
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=2 
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )
    
    print(f"Found {len(full_dataset)} images in {len(config.CLASSES)} classes.")
    print(f"Split into {len(train_dataset)} training images and {len(val_dataset)} validation images.")

    return train_loader, val_loader


if __name__ == '__main__':
    train_dataloader, val_dataloader = get_dataloaders()

    # Let's check a batch to see if it works
    for images, labels in train_dataloader:
        print("Batch of images has shape: ", images.shape) # [batch_size, 3, IMG_SIZE, IMG_SIZE]
        print("Batch of labels has shape: ", labels.shape) # [batch_size]
        break