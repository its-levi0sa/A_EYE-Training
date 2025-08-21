import torch
import torchvision.transforms as T
from PIL import Image
import argparse
import numpy as np
import os

# --- Important: Ensure your model files are accessible ---
from model.aeye_model import AEyeModel

def generate_explanation(tokens):
    """
    Converts the raw token tensor into a human-readable explanation.
    This version is specifically designed to handle and analyze 8 rings.
    """
    # Squeeze the batch dimension and move tensor to CPU for numpy conversion
    # Expected shape: [1, 8, 9] -> [8, 9]
    tokens = tokens.squeeze(0).cpu().numpy()
    num_rings = tokens.shape[0]

    # --- Start building the report string ---
    explanation = "Explainability Report (Based on 8-Ring Radial Token Analysis):\n"
    explanation += "-----------------------------------------------------------------\n"

    # --- Heuristic-Based Overall Assessment ---
    avg_brightness = np.mean(tokens[:, 0:3])
    avg_variation = np.mean(tokens[:, 3:6])
    
    # The "core" is defined as the inner quarter of the rings (rings 1-2 for an 8-ring model)
    core_ring_count = num_rings // 4
    core_brightness = np.mean(tokens[0:core_ring_count, 0:3])

    # Calculate proxies for clinical metrics
    coverage_proxy = min(100.0, (avg_brightness / 160.0) * 100)
    variation_based_opacity = (avg_variation / 50.0) * 100

    # Add a bonus to opacity if the core is exceptionally bright (dense nuclear sclerosis)
    brightness_bonus = 0
    if core_brightness > 190:
        brightness_bonus = ((core_brightness - 190) / (255 - 190)) * 40
    
    opacity_proxy = min(100.0, variation_based_opacity + brightness_bonus)

    explanation += f"Estimated Pupillary Coverage (Proxy): {coverage_proxy:.1f}%\n"
    explanation += f"Estimated Opacity (Proxy): {opacity_proxy:.1f}%\n\n"

    # --- Detailed Ring Zone Analysis for 8 Rings ---
    explanation += "Ring Zone Analysis:\n"

    # For 8 rings, it's logical to group them into two main zones.
    zone_names = ["Core & Inner Zone (Rings 1-4)", "Outer & Peripheral Zone (Rings 5-8)"]
    rings_per_zone = num_rings // 2

    for i, zone_name in enumerate(zone_names):
        start_index = i * rings_per_zone
        end_index = start_index + rings_per_zone
        zone_tokens = tokens[start_index:end_index]

        # Calculate average brightness and texture variation for the zone
        mean_brightness = zone_tokens[:, 0:3].mean()
        std_dev = zone_tokens[:, 3:6].mean()

        explanation += f"  - {zone_name}:\n"
        explanation += f"    - Avg. Brightness: {mean_brightness:.2f}\n"
        explanation += f"    - Avg. Color Variation (Texture): {std_dev:.2f}\n"

    explanation += "-----------------------------------------------------------------\n"
    return explanation


def predict(config):
    """
    Loads the trained 8-ring model and makes a prediction on a single image,
    providing a classification, confidence score, and detailed explanation.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Load Model Architecture ---
    model = AEyeModel(config['model_config'])
    
    # --- 2. Load Trained Weights ---
    try:
        model.load_state_dict(torch.load(config['model_path'], map_location=device))
        print(f"Successfully loaded model weights from {config['model_path']}")
    except FileNotFoundError:
        print(f"ERROR: Model file not found at '{config['model_path']}'. Please check the path.")
        return
    except Exception as e:
        print(f"ERROR: An error occurred while loading the model: {e}")
        return

    model.to(device)
    model.eval()

    # --- 3. Load and Preprocess the Input Image ---
    try:
        image = Image.open(config['image_path']).convert("RGB")
    except FileNotFoundError:
        print(f"ERROR: Image file not found at '{config['image_path']}'. Please check the path.")
        return

    # Define the same transformations used during validation/testing.
    data_transforms = T.Compose([
        T.Resize((128, 128)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Add a batch dimension (from [C, H, W] to [1, C, H, W]) and send to device.
    input_tensor = data_transforms(image).unsqueeze(0).to(device)

    # --- 4. Make Prediction and Extract Tokens ---
    with torch.no_grad(): # Disable gradient calculations for inference
        output, tokens = model(input_tensor, return_tokens=True)

    # --- 5. Process and Display Results ---
    # Apply sigmoid to the raw logit output to get a probability score
    probability = torch.sigmoid(output).item()
    prediction = "Mature" if probability >= 0.5 else "Immature"

    print(f"\n--- Prediction Results for: {os.path.basename(config['image_path'])} ---")
    print(f"Final Classification: {prediction}")
    print(f"Model Confidence Score: {probability:.4f} ({probability*100:.2f}%)")

    # --- 6. Generate and Print the Explanation ---
    explanation_report = generate_explanation(tokens)
    print(explanation_report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run 8-Ring A-EYE model for prediction and explanation.")
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image file.')
    parser.add_argument('--model_path', type=str, default='saved_models/aeye_best_model.pth', help='Path to the trained .pth model file for the 8-ring model.')
    args = parser.parse_args()

    # --- Model Configuration ---
    model_config = {
        'dims': [16, 32, 96, 128],
        'embed_dim': 192,
    }

    # --- Run Configuration ---
    run_config = {
        'model_config': model_config,
        'image_path': args.image_path,
        'model_path': args.model_path
    }

    predict(run_config)