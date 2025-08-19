import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import argparse
import logging
from tqdm import tqdm
import glob
import random

# Import custom modules
from model.aeye_model import AEyeModel
from data.dataset import CataractDataset

def train_model(config):
    """Main function to run the training and evaluation pipeline."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # --- Data Loading ---
    # 1. Define the paths to data directories
    train_dir = 'data/train'
    val_dir = 'data/val'

    # 2. Define classes and assign integer labels
    class_map = {
        'immature': 0,
        'mature': 1
    }

    # 3. Create a function to scan directories and get file paths with labels
    def get_paths_and_labels(data_dir, class_mapping):
        all_paths = []
        all_labels = []
        # Find all image files (jpg, jpeg, png) in the subdirectories
        for class_name, label in class_mapping.items():
            class_path = os.path.join(data_dir, class_name)
            image_paths = glob.glob(os.path.join(class_path, '*.[jp][pn]g'))
            all_paths.extend(image_paths)
            all_labels.extend([label] * len(image_paths))
            
        return all_paths, all_labels

    # Get the training and validation data by calling the function
    train_paths, train_labels = get_paths_and_labels(train_dir, class_map)
    val_paths, val_labels = get_paths_and_labels(val_dir, class_map)

    # 4. Shuffle the training data to ensure randomness
    temp_train_data = list(zip(train_paths, train_labels))
    random.shuffle(temp_train_data)
    train_paths, train_labels = zip(*temp_train_data)

    # 5. Print a summary to confirm that the data was loaded correctly
    logging.info(f"Found {len(train_paths)} images for training.")
    logging.info(f"Found {len(val_paths)} images for validation.")

    data_transforms = T.Compose([
        T.Resize((128, 128)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_dataset = CataractDataset(train_paths, train_labels, transform=data_transforms)
    val_dataset = CataractDataset(val_paths, val_labels, transform=data_transforms)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=2)
    logging.info("Data loaders created.")

    # --- Model, Optimizer, Loss ---
    model = AEyeModel(config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=0.01)

    # Calculate the weight to handle the imbalanced dataset.
    num_mature = len(glob.glob(os.path.join('data/train', 'mature', '*')))
    num_immature = len(glob.glob(os.path.join('data/train', 'immature', '*')))

    # The weight is the ratio of the majority class to the minority class.
    if num_immature > 0:
        weight = num_mature / num_immature
    else:
        weight = 1.0

    # 'immature' is label 0 and 'mature' is label 1.
    # Applying the weight to the 'mature' class.
    pos_weight = torch.tensor([weight], device=device)
    logging.info(f"Using a weighted loss. Weight for 'mature' class: {weight:.2f}")

    # Pass the calculated weight to the loss function.
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])

    logging.info("Model, optimizer, and loss function initialized.")
    logging.info(f"Model Configuration: {config}")

    best_val_f1 = 0.0
    os.makedirs(config['save_dir'], exist_ok=True)

    # --- Training Loop ---
    for epoch in range(config['epochs']):
        model.train()
        train_loop = tqdm(train_loader, leave=True, desc=f"Epoch {epoch+1}/{config['epochs']}")

        for inputs, labels in train_loop:
            inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loop.set_postfix(loss=loss.item())

        scheduler.step()

        # --- Validation Loop ---
        model.eval()
        val_preds, val_labels_all = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                preds = torch.sigmoid(outputs) > 0.5
                val_preds.extend(preds.cpu().numpy().flatten())
                val_labels_all.extend(labels.cpu().numpy().flatten())

        accuracy = accuracy_score(val_labels_all, val_preds)
        precision = precision_score(val_labels_all, val_preds, zero_division=0)
        recall = recall_score(val_labels_all, val_preds, zero_division=0)
        f1 = f1_score(val_labels_all, val_preds, zero_division=0)
        logging.info(f"Validation - Acc: {accuracy:.4f}, P: {precision:.4f}, R: {recall:.4f}, F1: {f1:.4f}")

        if f1 > best_val_f1:
            best_val_f1 = f1
            model_path = os.path.join(config['save_dir'], 'aeye_best_model.pth')
            torch.save(model.state_dict(), model_path)
            logging.info(f"New best model saved to {model_path} with F1-Score: {f1:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train A-EYE Cataract Classification Model")
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Optimizer learning rate')
    parser.add_argument('--save_dir', type=str, default='saved_models', help='Directory to save models')
    args = parser.parse_args()

    config = {
        'dims': [16, 32, 96, 128],
        'embed_dim': 192,
    }
    config.update(vars(args))

    train_model(config)