import torch
import cv2
import argparse
import numpy as np
import os
import glob
import albumentations as A
from albumentations.pytorch import ToTensorV2
import logging
import warnings
from model.aeye_model import AEyeModel

warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
MIN_MODELS_FOR_ENSEMBLE = 3
MATURE_COVERAGE_THRESHOLD = 80.0

def get_transforms():
    """Returns the same validation transforms used during training."""
    return A.Compose([
        A.Resize(256, 256),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
    ])

def generate_explanation(tokens_tensor, num_rings):
    """
    Generates a human-readable report and returns the key heuristic values.
    """
    if tokens_tensor is None or tokens_tensor.numel() == 0:
        return "Explainability report could not be generated (no token data).", {}

    avg_tokens = tokens_tensor.mean(dim=0).squeeze(0).cpu().numpy()

    # --- Denormalize token values to revert them to the [0, 255] pixel scale ---
    denormalized_tokens = np.zeros_like(avg_tokens)
    denormalized_tokens[:, 0:3] = (avg_tokens[:, 0:3] * 0.5 + 0.5) * 255
    denormalized_tokens[:, 3:6] = (avg_tokens[:, 3:6] * 0.5) * 255
    denormalized_tokens[:, 6:9] = (avg_tokens[:, 6:9] * 0.5 + 0.5) * 255

    explanation = "\n--- Heuristic Explainability Report ---\n"
    explanation += f"(Based on {num_rings}-Ring Token Analysis from Ensemble Average)\n"
    explanation += "------------------------------------------------------\n"
    explanation += "DISCLAIMER: This is a heuristic interpretation of the model's internal data, not a clinical diagnosis.\n\n"

    # --- Heuristic Calculations ---
    avg_brightness = np.mean(denormalized_tokens[:, 0:3])
    avg_variation = np.mean(denormalized_tokens[:, 3:6])
    
    core_ring_count = max(1, num_rings // 4)
    core_brightness = np.mean(denormalized_tokens[0:core_ring_count, 0:3])

    coverage_proxy = min(100.0, (avg_brightness / 160.0) * 100)
    variation_based_opacity = (avg_variation / 50.0) * 100
    brightness_bonus = 0
    if core_brightness > 190:
        brightness_bonus = ((core_brightness - 190) / (255 - 190)) * 40
    opacity_proxy = min(100.0, variation_based_opacity + brightness_bonus)

    explanation += f"Estimated Pupillary Coverage (Proxy): {coverage_proxy:.1f}%\n"
    explanation += f"Estimated Opacity (Proxy): {opacity_proxy:.1f}%\n\n"
    explanation += "Zonal Analysis:\n"
    
    # --- Dynamic Zone Definitions ---
    if num_rings == 4:
        zone_definitions = {"Core (Ring 1)": (0,1), "Inner (Ring 2)": (1,2), "Outer (Ring 3)": (2,3), "Peripheral (Ring 4)": (3,4)}
    elif num_rings == 8:
        zone_definitions = {"Core/Inner (Rings 1-4)": (0, 4), "Outer/Peripheral (Rings 5-8)": (4, 8)}
    else:
        zone_definitions = {"Core (Rings 1-4)": (0, 4), "Inner (Rings 5-8)": (4, 8), "Outer (Rings 9-12)": (8, 12), "Peripheral (Rings 13-16)": (12, 16)}

    for zone_name, (start, end) in zone_definitions.items():
        zone_tokens = denormalized_tokens[start:end]
        mean_brightness = zone_tokens[:, 0:3].mean()
        std_dev = zone_tokens[:, 3:6].mean()
        explanation += f"  - {zone_name}:\n"
        explanation += f"    - Avg. Brightness: {mean_brightness:.2f}\n"
        explanation += f"    - Avg. Color Variation: {std_dev:.2f}\n"
        
    heuristic_values = {
        "coverage_proxy": coverage_proxy,
        "opacity_proxy": opacity_proxy,
    }
    return explanation, heuristic_values

def predict_with_ensemble(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_paths = glob.glob(os.path.join(config['model_dir'], 'aeye_best_model_fold_*.pth'))
    if not model_paths:
        logging.error(f"No models found in '{config['model_dir']}'. Please check the directory path.")
        return

    models = []
    num_rings = None
    for path in model_paths:
        try:
            model = AEyeModel(config['model_config']).to(device)
            model.load_state_dict(torch.load(path, map_location=device))
            model.eval()
            models.append(model)

            if num_rings is None:
                num_rings = model.num_rings
        except Exception as e:
            logging.warning(f"Could not load model from {path}. Skipping. Error: {e}")
    
    if len(models) < MIN_MODELS_FOR_ENSEMBLE:
        logging.warning(f"Only {len(models)} model(s) loaded. An ensemble prediction requires at least {MIN_MODELS_FOR_ENSEMBLE} for reliability.")
        if not models: return

    logging.info(f"Loaded {len(models)} models for ensembling.")

    try:
        image = cv2.imread(config['image_path'])
        if image is None: raise FileNotFoundError("Image not found or could not be read.")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception as e:
        logging.error(f"Failed to load image at '{config['image_path']}'. Error: {e}")
        return

    transforms = get_transforms()
    input_tensor = transforms(image=image)['image'].unsqueeze(0).to(device)

    all_tokens = []
    with torch.no_grad():
        for model in models:
            _, tokens = model(input_tensor, return_tokens=True)
            all_tokens.append(tokens)

    # --- RULE-BASED PREDICTION LOGIC ---
    if all_tokens:
        stacked_tokens = torch.stack(all_tokens, dim=0)
        explanation_report, heuristic_values = generate_explanation(stacked_tokens, num_rings)
        
        pupillary_coverage = heuristic_values.get("coverage_proxy", 0.0)

        prediction = "Mature" if pupillary_coverage >= MATURE_COVERAGE_THRESHOLD else "Immature"
        
        confidence_score = pupillary_coverage / 100.0
        
        print(f"\n--- Ensemble Prediction for {os.path.basename(config['image_path'])} ---")
        print(f"Final Classification: {prediction}")
        print(f"Confidence (based on Pupillary Coverage): {confidence_score:.4f} ({pupillary_coverage:.2f}%)")
        print(explanation_report)
    else:
        print("Could not generate a prediction as no token data was returned by the models.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run A-EYE model ensemble for prediction on a single image.")
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