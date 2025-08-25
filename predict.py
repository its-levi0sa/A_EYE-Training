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
    return A.Compose([
        A.Resize(256, 256),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
    ])

def generate_explanation(tokens):
    """
    Generates a human-readable report from the 4 radial tokens with cleaner formatting.
    """
    if tokens is None:
        return "Explainability report could not be generated."
        
    tokens = tokens.squeeze(0).cpu().numpy()
    
    # --- Start building the report string with cleaner indentation ---
    explanation = "Explainability Report (Based on Radial Token Analysis):\n"
    explanation += "------------------------------------------------------\n"

    # --- Heuristic-Based Overall Assessment ---
    avg_brightness = np.mean(tokens[:, 0:3])
    avg_variation = np.mean(tokens[:, 3:6])
    core_brightness = np.mean(tokens[0, 0:3])

    coverage_proxy = min(100.0, (avg_brightness / 160.0) * 100)
    variation_based_opacity = (avg_variation / 50.0) * 100
    brightness_bonus = 0
    if core_brightness > 190:
        brightness_bonus = ((core_brightness - 190) / (255 - 190)) * 40
    opacity_proxy = min(100.0, variation_based_opacity + brightness_bonus)

    explanation += f"Estimated Pupillary Coverage (Proxy): {coverage_proxy:.1f}%\n"
    explanation += f"Estimated Opacity (Proxy): {opacity_proxy:.1f}%\n"
    explanation += "Ring Zone Analysis:\n"
    
    # --- Detailed Ring-by-Ring Analysis ---
    ring_definitions = ["Core Zone", "Inner Zone", "Outer Zone", "Peripheral Zone"]
    for i, ring_name in enumerate(ring_definitions):
        ring_token = tokens[i]
        mean_brightness = ring_token[0:3].mean()
        std_dev = ring_token[3:6].mean()
        # Using a single level of indentation for sub-points
        explanation += f"  - {ring_name}:\n"
        explanation += f"    - Avg. Brightness: {mean_brightness:.2f}\n"
        explanation += f"    - Avg. Color Variation: {std_dev:.2f}\n"
        
    return explanation

def predict_with_ensemble(config):
    """
    Loads all K-Fold models, runs prediction with each, and averages the results.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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

    try:
        image = cv2.imread(config['image_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"ERROR loading image: {e}")
        return

    transforms = get_transforms()
    augmented = transforms(image=image)
    input_tensor = augmented['image'].unsqueeze(0).to(device)

    all_probabilities = []
    first_model_tokens = None
    with torch.no_grad():
        for i, model in enumerate(models):
            if i == 0:
                output, tokens = model(input_tensor, return_tokens=True)
                first_model_tokens = tokens
            else:
                output = model(input_tensor, return_tokens=False)

            probability = torch.sigmoid(output).item()
            all_probabilities.append(probability)
            logging.info(f"Model {i+1} prediction: {probability:.4f}")

    final_probability = np.mean(all_probabilities)
    prediction = "Mature" if final_probability >= 0.5 else "Immature"
    
    print(f"\n--- Prediction Results for {os.path.basename(config['image_path'])} ---")
    print(f"Final Classification: {prediction}")
    print(f"Model Confidence Score: {final_probability:.4f} ({final_probability*100:.2f}%)")
    
    explanation_report = generate_explanation(first_model_tokens)
    print(explanation_report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run A-EYE model ensemble for prediction.")
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image file.')
    parser.add_argument('--model_dir', type=str, default='saved_models', help='Directory containing the trained K-Fold model files.')
    args = parser.parse_args()

    model_config = {
        'dims': [32, 64, 128, 160],
        'embed_dim': 256,
    }

    config = {'model_config': model_config}
    config.update(vars(args))
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    predict_with_ensemble(config)