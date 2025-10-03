import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm
import copy

from src import config
from src.data_preprocessing import get_dataloaders

def train_model():
    """
    Main function to train the image classification model.
    """

    device = torch.device(config.DEVICE)
    print(f"Using device: {device}")

    train_loader, val_loader = get_dataloaders()

    # Load a pre-trained MobileNetV2 model
    model = models.mobilenet_v2(weights='MobileNet_V2_Weights.DEFAULT')

    # Freeze all the parameters in the feature extraction part of the model
    for param in model.parameters():
        param.requires_grad = False

    # Replace the final classification layer with a new one for our number of classes
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, config.NUM_CLASSES)

    
    model = model.to(device)

    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier[1].parameters(), lr=config.LEARNING_RATE)
    
    # Training loop
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    print("Starting model training...")
    for epoch in range(config.NUM_EPOCHS):
        print(f'Epoch {epoch+1}/{config.NUM_EPOCHS}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                dataloader = train_loader
            else:
                model.eval()   # Set model to evaluate mode
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data using a progress bar
            progress_bar = tqdm(dataloader, desc=f"{phase.capitalize()} Phase")
            for inputs, labels in progress_bar:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                # Track history only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward pass + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                progress_bar.set_postfix(loss=loss.item())

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep copy the model if we have a new best validation accuracy
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                # Save the best model
                torch.save(model.state_dict(), config.MODEL_PATH)
                print(f"New best model saved to {config.MODEL_PATH} with accuracy: {best_acc:.4f}")

    print(f'Best val Acc: {best_acc:4f}')
    print("Training complete.")

# Make the script executable
if __name__ == '__main__':
    train_model()