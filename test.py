import torch
import torch.optim as optim
import numpy as np
import logging
from sklearn.metrics import f1_score

from model.aeye_model import AEyeModel
from train import get_transforms, FocalLoss

def run_training_smoke_test():
    """
    Performs a smoke test of the entire 8-ring training pipeline.
    It simulates one training step (forward, backward, optimizer step) 
    and one validation step to ensure all components are connected.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("--- Starting Training Smoke Test for 8-RING MODEL ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    try:
        # 1. Initialize Model
        model_config = {'dims': [32, 64, 128, 160], 'embed_dim': 256}
        model = AEyeModel(model_config).to(device)
        logging.info("✅ Model initialized successfully.")

        # 2. Create Dummy Data and DataLoader
        dummy_image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        transforms = get_transforms(is_train=True)
        input_tensor = transforms(image=dummy_image)['image'].unsqueeze(0).to(device)
        dummy_labels = torch.ones(1, 1).to(device) # Batch size of 1
        logging.info(f"✅ Dummy data created. Shape: {input_tensor.shape}")

        # 3. Initialize Optimizer and Loss
        optimizer = optim.AdamW(model.parameters(), lr=1e-4)
        criterion = FocalLoss()
        logging.info("✅ Optimizer and Loss function initialized.")

        # --- 4. Test One Training Step ---
        logging.info("--> Testing a single training step...")
        model.train()
        optimizer.zero_grad()
        
        output = model(input_tensor)
        logging.info("    Forward pass successful.")
        
        loss = criterion(output, dummy_labels)
        logging.info(f"    Loss calculated: {loss.item():.4f}")

        loss.backward()
        logging.info("    Backward pass successful.")

        optimizer.step()
        logging.info("    Optimizer step successful.")
        logging.info("✅ Training step test passed!")

        # --- 5. Test One Validation Step ---
        logging.info("--> Testing a single validation step...")
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
            preds = torch.sigmoid(output) > 0.5
            f1 = f1_score(dummy_labels.cpu(), preds.cpu())
            logging.info(f"    Validation metrics calculated. F1: {f1:.4f}")
        logging.info("✅ Validation step test passed!")

    except Exception as e:
        logging.error("❌ TEST FAILED: An error occurred.", exc_info=True)
        return

    logging.info("--- ✅ 8-RING TRAINING SMOKE TEST PASSED! ---")

if __name__ == '__main__':
    run_training_smoke_test()