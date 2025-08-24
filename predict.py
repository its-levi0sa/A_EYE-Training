import torch
import torch.nn as nn
import cv2
import argparse
import numpy as np
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Import the primary model
from model.aeye_model import AEyeModel

def get_transforms():
    """
    Defines the transformations for a single prediction image.
    MUST match the validation transforms from the training script.
    """
    return A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
    ])

def generate_explanation(tokens):
    """
    Generates a human-readable report from the 4 radial tokens.
    """
    # Expected shape: [1, 4, 9] -> [4, 9]
    tokens = tokens.squeeze(0).cpu().numpy()
    
    explanation = "Explainability Report (4-Ring Model | 256x256)\n"
    explanation += "---------------------------------------------------\n"

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
    explanation += f"Estimated Opacity (Proxy): {opacity_proxy:.1f}%\n\n"

    # --- Detailed Ring-by-Ring Analysis ---
    explanation += "Ring-by-Ring Analysis (Center -> Periphery):\n"
    ring_definitions = ["Core (0-32px)", "Inner (32-64px)", "Outer (64-96px)", "Peripheral (96-128px)"]
    for i, ring_name in enumerate(ring_definitions):
        ring_token = tokens[i]
        mean_brightness = ring_token[0:3].mean()
        std_dev = ring_token[3:6].mean()
        explanation += f"  - Ring {i+1} [{ring_name}]:\n"
        explanation += f"    - Avg. Brightness: {mean_brightness:.2f}\n"
        explanation += f"    - Avg. Color Variation (Texture): {std_dev:.2f}\n"
    explanation += "---------------------------------------------------\n"
    return explanation

def predict(config):
    """Loads and runs the 4-ring model for a single prediction."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- 1. Load Model Architecture ---
    model = AEyeModel(config['model_config']).to(device)
    
    # --- 2. Load Trained Weights ---
    try:
        model.load_state_dict(torch.load(config['model_path'], map_location=device))
    except Exception as e:
        print(f"ERROR loading model weights: {e}")
        return

    model.eval()

    # --- 3. Load and Preprocess Image ---
    try:
        image = cv2.imread(config['image_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"ERROR loading image: {e}")
        return

    transforms = get_transforms()
    augmented = transforms(image=image)
    input_tensor = augmented['image'].unsqueeze(0).to(device)

    # --- 4. Make Prediction ---
    with torch.no_grad():
        # Ensure model's forward pass can return tokens
        if 'return_tokens' in AEyeModel.forward.__code__.co_varnames:
             output, tokens = model(input_tensor, return_tokens=True)
        else:
            print("WARNING: Model's forward pass does not support 'return_tokens'. Cannot generate explanation.")
            output = model(input_tensor)
            tokens = None

    # --- 5. Display Results ---
    probability = torch.sigmoid(output).item()
    prediction = "Mature" if probability >= 0.5 else "Immature"

    print(f"\n--- Prediction Results for: {os.path.basename(config['image_path'])} ---")
    print(f"Final Classification: {prediction}")
    print(f"Model Confidence Score: {probability:.4f} ({probability*100:.2f}%)")

    if tokens is not None:
        explanation_report = generate_explanation(tokens)
        print(explanation_report)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run 4-Ring A-EYE model (256x256) for prediction.")
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image file.')
    parser.add_argument('--model_path', type=str, default='saved_models/aeye_best_model_4_rings.pth', help='Path to the trained 4-ring .pth model file.')
    args = parser.parse_args()

    model_config = {
        'dims': [16, 32, 96, 128],
        'embed_dim': 192,
    }
    
    config = {'model_config': model_config}
    config.update(vars(args))
    predict(config)
