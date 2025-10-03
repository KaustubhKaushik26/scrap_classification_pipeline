import torch

# Project Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED = 42

# Dataset Configuration
DATA_DIR = "data"
CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
NUM_CLASSES = len(CLASSES)

# Model & Training Configuration
MODEL_NAME = "mobilenet_v2"
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 10
IMG_SIZE = 224 # Input size for MobileNetV2

# File Paths
MODEL_PATH = "models/mobilenet_v2_scrap_classifier.pth"
ONNX_MODEL_PATH = "models/mobilenet_v2_scrap_classifier.onnx"
RESULTS_CSV_PATH = "results/classification_log.csv"

# Simulation Configuration 
CONFIDENCE_THRESHOLD = 0.70

# Plot config
CLASS_DISTRIBUTION_PATH = "results/class_distribution.png"
CONFUSION_MATRIX_PATH = "results/confusion_matrix.png"