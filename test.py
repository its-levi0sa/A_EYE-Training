import torch
import torch.utils.data
import torchvision.transforms as T
import logging
import glob

# Import primary modules
from model.aeye_model import AEyeModel
from data.dataset import CataractDataset

def run_real_data_test():
    """
    Performs a test run using a small batch of actual images to verify the
    entire data loading and model pipeline.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    logging.info("--- Starting Test with Real Data ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    try:
        # --- 1. Find and Load Real Images ---
        logging.info("Searching for test images in 'test_images/' directory...")
        mature_paths = glob.glob("C:/Users/denni/Downloads/mature/*.png")
        immature_paths = glob.glob("C:/Users/denni/Downloads/immature/*.png")

        if not mature_paths or not immature_paths:
            logging.error("❌ TEST FAILED: Could not find images in 'test_images/mature' or 'test_images/immature'.")
            logging.error("Please create the folders and add a few sample images to test.")
            return

        test_paths = mature_paths + immature_paths
        test_labels = [1] * len(mature_paths) + [0] * len(immature_paths)
        logging.info(f"Found {len(test_paths)} total test images.")

        # --- 2. Create Dataset and DataLoader ---
        data_transforms = T.Compose([
            T.Resize((128, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        test_dataset = CataractDataset(test_paths, test_labels, transform=data_transforms)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4)
        logging.info("✅ Dataset and DataLoader created successfully.")

        # --- 3. Initialize Model ---
        config = {'dims': [16, 32, 96, 128], 'embed_dim': 192}
        model = AEyeModel(config).to(device)
        logging.info("✅ AEyeModel initialized successfully.")

        # --- 4. Process One Batch ---
        logging.info("Fetching one batch of real data...")
        images, labels = next(iter(test_loader))
        images = images.to(device)
        logging.info(f"✅ Batch loaded successfully. Batch shape: {images.shape}")

        logging.info("Performing a forward pass with real data...")
        output = model(images)
        logging.info("✅ Forward pass completed successfully!")
        logging.info(f"Output tensor shape: {output.shape}")

        assert output.shape[0] == len(images), "Output batch size does not match input."
        logging.info("✅ Output shape is correct.")

    except Exception as e:
        logging.error(f"❌ TEST FAILED: An error occurred during the test.")
        logging.error(f"Details: {e}", exc_info=True) # exc_info gives more details
        return

    logging.info("--- ✅ Real Data Test Passed! ---")
    logging.info("Your entire pipeline is working correctly with your actual images.")

if __name__ == '__main__':
    run_real_data_test()