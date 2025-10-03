import torch
from torchvision import models
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from src import config
from src.data_preprocessing import get_dataloaders

def evaluate_model():
    """
    Evaluates the trained model on the validation set and generates a
    classification report and a confusion matrix plot.
    """
    print("Starting model evaluation...")
    device = torch.device(config.DEVICE)

    # We only need the validation dataloader for evaluation
    _, val_loader = get_dataloaders()

    
    # Re-create the model structure
    model = models.mobilenet_v2()
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(num_ftrs, config.NUM_CLASSES)
    
   
    model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))
    model.to(device)
    model.eval() # Set model to evaluation mode

    
    all_preds = []
    all_labels = []

    with torch.no_grad(): 
        for inputs, labels in tqdm(val_loader, desc="Running predictions"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Classification Report
    print("\n" + "="*50)
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=config.CLASSES))
    print("="*50)

    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=config.CLASSES, yticklabels=config.CLASSES)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    
    # Save the plot
    plt.savefig(config.CONFUSION_MATRIX_PATH)
    print(f"Confusion matrix plot saved to: {config.CONFUSION_MATRIX_PATH}")
    # plt.show() # Uncomment to display the plot directly

if __name__ == '__main__':
    evaluate_model()