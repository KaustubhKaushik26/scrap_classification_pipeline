import torch
from torchvision import models
import os
from src import config

def export_model_to_onnx():
    """
    Loads the trained model and exports it to ONNX format.
    """
    print("Exporting model to ONNX...")
    
    device = torch.device(config.DEVICE)

    model = models.mobilenet_v2()
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(num_ftrs, config.NUM_CLASSES)

    # Load the saved weights
    model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))
    model.to(device)
    model.eval() # Set the model to evaluation mode

    # Create a dummy input tensor with the correct shape
    dummy_input = torch.randn(1, 3, config.IMG_SIZE, config.IMG_SIZE, device=device)

    # Export the model
    torch.onnx.export(model,
                      dummy_input,
                      config.ONNX_MODEL_PATH,
                      export_params=True,
                      opset_version=11,
                      do_constant_folding=True,
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})

    print(f"Model successfully exported to {config.ONNX_MODEL_PATH}")

if __name__ == '__main__':
    export_model_to_onnx()