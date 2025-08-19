import torch
import torchvision.transforms as T
from PIL import Image
import argparse
import numpy as np

# Import main model
from model.aeye_model import AEyeModel

def generate_explanation(tokens):
    """
    Converts the raw token tensor into a human-readable explanation.
    This version is updated to dynamically handle 16 rings.
    """
    # Tokens shape: [1, 16, 9] -> [16, 9]
    tokens = tokens.squeeze(0).cpu().numpy()
    num_rings = tokens.shape[0]
    
    explanation = "Explainability Report (Based on Radial Token Analysis):\n"
    explanation += "------------------------------------------------------\n"
    
    # --- Heuristics ---
    avg_brightness = np.mean(tokens[:, 0:3]) 
    avg_variation = np.mean(tokens[:, 3:6])
    core_ring_count = num_rings // 4
    core_brightness = np.mean(tokens[0:core_ring_count, 0:3])

    coverage_proxy = min(100, (avg_brightness / 160.0) * 100)
    variation_based_opacity = (avg_variation / 50.0) * 100
    
    brightness_bonus = 0
    if core_brightness > 190:
        brightness_bonus = ((core_brightness - 190) / (255 - 190)) * 40
        
    opacity_proxy = min(100, variation_based_opacity + brightness_bonus)

    explanation += f"Estimated Pupillary Coverage (Proxy): {coverage_proxy:.1f}%\n"
    explanation += f"Estimated Opacity (Proxy): {opacity_proxy:.1f}%\n\n"
    
    # --- Updated Ring-by-Ring Analysis for 16 Rings ---
    explanation += "Ring Zone Analysis:\n"
    
    # Define the four main zones
    zone_names = ["Core Zone (Rings 1-4)", "Inner Zone (Rings 5-8)", "Outer Zone (Rings 9-12)", "Peripheral Zone (Rings 13-16)"]
    rings_per_zone = num_rings // 4

    for i, zone_name in enumerate(zone_names):
        start_index = i * rings_per_zone
        end_index = start_index + rings_per_zone
        zone_tokens = tokens[start_index:end_index]
        
        mean_brightness = zone_tokens[:, 0:3].mean()
        std_dev = zone_tokens[:, 3:6].mean()
        
        explanation += f"  - {zone_name}:\n"
        explanation += f"    - Avg. Brightness: {mean_brightness:.2f}\n"
        explanation += f"    - Avg. Color Variation: {std_dev:.2f}\n"
        
    explanation += "------------------------------------------------------\n"
    return explanation


def predict(config):
    """Loads the trained model and makes a prediction on a single image."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 1. Load Model ---
    model = AEyeModel(config['model_config'])
    try:
        model.load_state_dict(torch.load(config['model_path'], map_location=device))
    except FileNotFoundError:
        print(f"ERROR: Model file not found at {config['model_path']}")
        return
        
    model.to(device)
    model.eval()

    # --- 2. Load and Preprocess Image ---
    try:
        image = Image.open(config['image_path']).convert("RGB")
    except FileNotFoundError:
        print(f"ERROR: Image file not found at {config['image_path']}")
        return
    
    data_transforms = T.Compose([
        T.Resize((128, 128)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    input_tensor = data_transforms(image).unsqueeze(0).to(device)

    # --- 3. Make Prediction and Get Tokens ---
    with torch.no_grad():
        output, tokens = model(input_tensor, return_tokens=True) 
        
    # --- 4. Process Output ---
    probability = torch.sigmoid(output).item()
    prediction = "Mature" if probability >= 0.5 else "Immature"

    print(f"\n--- Prediction Results for {config['image_path']} ---")
    print(f"Final Classification: {prediction}")
    print(f"Model Confidence Score: {probability:.4f}")
    
    # --- 5. Generate and Print Explanation ---
    explanation_report = generate_explanation(tokens)
    print(explanation_report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run A-EYE model for prediction and explanation.")
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image.')
    parser.add_argument('--model_path', type=str, default='saved_models/aeye_best_model.pth', help='Path to the trained .pth model file.')
    args = parser.parse_args()

    model_config = {
        'dims': [16, 32, 96, 128],
        'embed_dim': 192,
    }

    run_config = {
        'model_config': model_config,
        'image_path': args.image_path,
        'model_path': args.model_path
    }
    
    predict(run_config)