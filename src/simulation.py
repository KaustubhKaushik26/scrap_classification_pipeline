import os
import time
import random
import pandas as pd
from tqdm import tqdm
from src.inference import ONNXPredictor
from src import config

def run_simulation(num_images=50):
    """
    Simulates a real-time conveyor belt classification process.

    Args:
        num_images (int): The number of images to simulate.
    """
    print("Initializing simulation...")
    predictor = ONNXPredictor(config.ONNX_MODEL_PATH)
    
    # Prepare a list of images to simulate 
    all_image_paths = []
    for class_name in os.listdir(config.DATA_DIR):
        class_dir = os.path.join(config.DATA_DIR, class_name)
        if os.path.isdir(class_dir):
            for image_name in os.listdir(class_dir):
                all_image_paths.append(os.path.join(class_dir, image_name))
    
    # Randomly select a subset of images for the simulation
    random.shuffle(all_image_paths)
    simulation_images = all_image_paths[:num_images]

    # Main Simulation Loop 
    results = []
    print(f"\nStarting conveyor belt simulation for {num_images} items...")
    
    for image_path in tqdm(simulation_images, desc="Simulating Conveyor Belt"):
        # 1. Classify the image
        predicted_class, confidence = predictor.predict(image_path)
        
        # 2. Log output to console
        print(f"\nItem: {os.path.basename(image_path)}")
        print(f"  -> Predicted Class: {predicted_class}, Confidence: {confidence:.2f}")

        # 3. Check against confidence threshold
        low_confidence_flag = confidence < config.CONFIDENCE_THRESHOLD
        if low_confidence_flag:
            print(f"  -> !!! LOW CONFIDENCE DETECTED !!!")

        # 4. Store in results list
        results.append({
            "timestamp": pd.Timestamp.now(),
            "image_path": image_path,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "low_confidence_flag": low_confidence_flag
        })
        
        # Simulate time between items on the conveyor
        time.sleep(0.5)

    # --- Save results to CSV ---
    results_df = pd.DataFrame(results)
    results_df.to_csv(config.RESULTS_CSV_PATH, index=False)
    
    print("\n" + "="*50)
    print("Simulation complete!")
    print(f"Results have been saved to: {config.RESULTS_CSV_PATH}")
    print("="*50)

if __name__ == '__main__':
    run_simulation(num_images=50) 