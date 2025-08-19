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
    This is where you can define what "high coverage" or "high opacity" means
    based on the statistical features.
    """
    # Tokens shape: [1, 4, 9] -> [4, 9]
    tokens = tokens.squeeze(0).cpu().numpy()
    
    # The 9 features are [mean_r, mean_g, mean_b, std_r, std_g, std_b, med_r, med_g, med_b]
    # High standard deviation (std) in color indicate opacity (less uniform color).
    # High mean brightness (mean of RGB) indicate a dense, white cataract (coverage).
    explanation = "Explainability Report (Based on Radial Token Analysis):\n"
    explanation += "------------------------------------------------------\n"
    
    ring_names = ["Innermost Ring (Core)", "Inner Ring", "Outer Ring", "Outermost Ring (Periphery)"]
    
    # Calculate overall metrics from tokens
    # Average brightness across all rings (proxy for coverage)
    avg_brightness = np.mean(tokens[:, 0:3]) 
    # Average color variation across all rings (proxy for opacity)
    avg_variation = np.mean(tokens[:, 3:6])
    
    # Convert these to a rough percentage
    # These are heuristics and should be described as such.
    # Max possible brightness/variation is 255.
    coverage_proxy = min(100, (avg_brightness / 150) * 100)
    opacity_proxy = min(100, (avg_variation / 50) * 100)

    explanation += f"Estimated Pupillary Coverage (Proxy): {coverage_proxy:.1f}%\n"
    explanation += f"Estimated Opacity (Proxy): {opacity_proxy:.1f}%\n\n"
    explanation += "Ring-by-Ring Analysis:\n"

    for i, ring_name in enumerate(ring_names):
        ring_data = tokens[i]
        mean_brightness = ring_data[0:3].mean()
        std_dev = ring_data[3:6].mean()
        explanation += f"  - {ring_name}:\n"
        explanation += f"    - Avg. Brightness: {mean_brightness:.2f} (A high value suggests dense cataract)\n"
        explanation += f"    - Color Variation (Std Dev): {std_dev:.2f} (A high value suggests non-uniform opacity)\n"
        
    explanation += "------------------------------------------------------\n"
    return explanation


def predict(config):
    """Loads the trained model and makes a prediction on a single image."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 1. Load Model ---
    model = AEyeModel(config['model_config'])
    model.load_state_dict(torch.load(config['model_path'], map_location=device))
    model.to(device)
    model.eval()

    # --- 2. Load and Preprocess Image ---
    image = Image.open(config['image_path']).convert("RGB")
    
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