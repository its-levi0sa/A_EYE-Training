import torch
import torch.nn as nn
import cv2
import argparse
import numpy as np
import os
import glob
import albumentations as A
from albumentations.pytorch import ToTensorV2
import logging

# Suppress warnings for a cleaner output
import warnings
warnings.filterwarnings("ignore")

# Import the primary model
from model.aeye_model import AEyeModel

def get_transforms():
    """
    Defines the transformations for a single prediction image.
    MUST match the validation transforms from the training script.
    """
    # This transform now exactly matches the one in your K-Fold training script
    return A.Compose([
        A.Resize(256, 256),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
    ])

def predict_with_ensemble(config):
    """
    Loads all K-Fold models, runs prediction with each, and averages the results.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- 1. Find and Load All Fold Models ---
    model_paths = glob.glob(os.path.join(config['model_dir'], 'aeye_best_model_fold_*.pth'))
    if not model_paths:
        print(f"ERROR: No models found in '{config['model_dir']}'. Please check the path.")
        return

    models = []
    for path in model_paths:
        model = AEyeModel(config['model_config']).to(device)
        try:
            model.load_state_dict(torch.load(path, map_location=device))
            model.eval()
            models.append(model)
        except Exception as e:
            print(f"Warning: Could not load model from {path}. Skipping. Error: {e}")
    
    if not models:
        print("ERROR: Failed to load any valid models.")
        return
        
    logging.info(f"Loaded {len(models)} models for ensembling.")

    # --- 2. Load and Preprocess Image ---
    try:
        image = cv2.imread(config['image_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"ERROR loading image: {e}")
        return

    transforms = get_transforms()
    augmented = transforms(image=image)
    input_tensor = augmented['image'].unsqueeze(0).to(device)

    # --- 3. Make Prediction with Each Model ---
    all_probabilities = []
    with torch.no_grad():
        for i, model in enumerate(models):
            output = model(input_tensor)
            probability = torch.sigmoid(output).item()
            all_probabilities.append(probability)
            logging.info(f"Model {i+1} prediction: {probability:.4f}")

    # --- 4. Average the Results ---
    final_probability = np.mean(all_probabilities)
    prediction = "Mature" if final_probability >= 0.5 else "Immature"

    print(f"\n--- Ensemble Prediction Results for: {os.path.basename(config['image_path'])} ---")
    print(f"Individual Model Probabilities: {[f'{p:.2f}' for p in all_probabilities]}")
    print(f"Final Classification: {prediction}")
    print(f"Final Confidence Score: {final_probability:.4f} ({final_probability*100:.2f}%)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run A-EYE model ensemble for prediction.")
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image file.')
    parser.add_argument('--model_dir', type=str, default='saved_models', help='Directory containing the trained K-Fold model files.')
    args = parser.parse_args()

    # This model_config MUST match the one used for training
    model_config = {
        'dims': [32, 64, 128, 160],
        'embed_dim': 256,
    }

    config = {'model_config': model_config}
    config.update(vars(args))
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    predict_with_ensemble(config)