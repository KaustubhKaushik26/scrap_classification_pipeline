import onnxruntime
import numpy as np
from PIL import Image
from torchvision import transforms
from src import config

class ONNXPredictor:
    """
    A class to handle model loading and inference using ONNX Runtime.
    """
    def __init__(self, model_path):
        """
        Initializes the predictor with the ONNX model.
        
        Args:
            model_path (str): The path to the ONNX model file.
        """
        self.session = onnxruntime.InferenceSession(model_path)
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(config.IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image_path):
        """
        Predicts the class and confidence for a single image.

        Args:
            image_path (str): The path to the input image file.

        Returns:
            tuple: A tuple containing the predicted class name (str) and confidence score (float).
        """
        # 1. Load and preprocess the image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0) # Add batch dimension
        image_np = image_tensor.numpy()

        # 2. Run inference
        inputs = {'input': image_np}
        outputs = self.session.run(None, inputs)
        
        # 3. Process the output
        logits = outputs[0]
        # Apply softmax to get probabilities
        probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
        
        confidence = float(np.max(probabilities))
        predicted_class_idx = int(np.argmax(probabilities))
        predicted_class_name = config.CLASSES[predicted_class_idx]

        return predicted_class_name, confidence


if __name__ == '__main__':
    # Create a predictor instance
    predictor = ONNXPredictor(config.ONNX_MODEL_PATH)

    # You need a sample image to test. Let's find one.
    # Replace this with an actual path to an image in your data folder
    import os
    import random
    
    sample_class = random.choice(config.CLASSES)
    sample_image_folder = os.path.join(config.DATA_DIR, sample_class)
    sample_image_name = random.choice(os.listdir(sample_image_folder))
    test_image_path = os.path.join(sample_image_folder, sample_image_name)

    print(f"Testing with image: {test_image_path}")

    # Make a prediction
    predicted_class, confidence = predictor.predict(test_image_path)

    print(f"Predicted Class: {predicted_class}")
    print(f"Confidence: {confidence:.4f}")