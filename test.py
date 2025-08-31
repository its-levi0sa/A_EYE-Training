import torch
import numpy as np
import logging
import cv2

# Import the necessary components
from model.aeye_model import AEyeModel
from train import get_transforms

def run_smoke_test():
    """
    Performs a self-contained smoke test to verify the entire 16-ring model pipeline.
    It creates dummy data, applies the correct validation transforms, and runs a forward pass.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("--- Starting Smoke Test for 16-RING MODEL ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    try:
        # --- 1. Create Dummy Data in Memory ---
        logging.info("Creating dummy data for testing...")
        # Create a batch of 2 dummy images (one for each class label)
        dummy_images = [(np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8), 1),
                        (np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8), 0)]
        
        class MockDataset(torch.utils.data.Dataset):
            def __init__(self, data, transform):
                self.data = data
                self.transform = transform
            def __len__(self):
                return len(self.data)
            def __getitem__(self, idx):
                image, label = self.data[idx]
                if self.transform:
                    image = self.transform(image=image)['image']
                return image, torch.tensor(label, dtype=torch.float32)

        logging.info("✅ Dummy data created.")

        # --- 2. Use Correct Validation Transforms ---
        logging.info("Applying validation transforms from train.py...")
        val_transforms = get_transforms(is_train=False)
        test_dataset = MockDataset(dummy_images, transform=val_transforms)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=2)
        logging.info("✅ Transforms and DataLoader are working.")

        # --- 3. Initialize Model with Correct Config ---
        logging.info("Initializing the 16-ring model architecture...")
        model_config = {
            'dims': [32, 64, 128, 160],
            'embed_dim': 256,
        }
        model = AEyeModel(model_config).to(device)
        logging.info("✅ AEyeModel initialized successfully.")

        # --- 4. Process One Batch ---
        logging.info("Fetching one batch of test data...")
        images, labels = next(iter(test_loader))
        images = images.to(device)
        logging.info(f"Batch loaded successfully. Batch shape: {images.shape}")

        logging.info("Performing a forward pass...")
        output, tokens = model(images, return_tokens=True)
        logging.info("✅ Forward pass completed!")
        
        # --- 5. Verify Output Shapes ---
        assert output.shape == (2, 1), f"Expected output shape (2, 1), but got {output.shape}"
        assert tokens.shape == (2, 16, 9), f"Expected token shape (2, 16, 9) for 16 rings, but got {tokens.shape}"
        logging.info(f"✅ Output shape is correct: {output.shape}")
        logging.info(f"✅ Token shape is correct for 16 rings: {tokens.shape}")

    except Exception as e:
        logging.error(f"❌ TEST FAILED: An error occurred during the test.")
        logging.error(f"Details: {e}", exc_info=True)
        return

    logging.info("--- ✅ 16-RING SMOKE TEST PASSED! ---")
    logging.info("Your entire pipeline is working correctly.")

if __name__ == '__main__':
    run_smoke_test()