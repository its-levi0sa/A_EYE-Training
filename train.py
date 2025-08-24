import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import argparse
import logging
from tqdm import tqdm
import glob
import random
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Import the primary model
from model.aeye_model import AEyeModel

# Uses Albumentations
def get_transforms(is_train=True):
    if is_train:
        return A.Compose([
            A.Resize(256, 256),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.75),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=25, p=0.75),
            A.Blur(blur_limit=3, p=0.2),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(256, 256),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2(),
        ])

# Modified dataset class for Albumentations
class AlbumentationsDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        # Read image with OpenCV for Albumentations
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, label

def train_model(config):
    """Main function to run the standardized training and evaluation pipeline."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # --- Data Loading ---
    train_dir = 'data/train'
    val_dir = 'data/val'
    class_map = {'immature': 0, 'mature': 1}

    def get_paths_and_labels(data_dir, class_mapping):
        all_paths, all_labels = [], []
        for class_name, label in class_mapping.items():
            class_path = os.path.join(data_dir, class_name)
            image_paths = glob.glob(os.path.join(class_path, '*.[jp][pn]g'))
            all_paths.extend(image_paths)
            all_labels.extend([label] * len(image_paths))
        return all_paths, all_labels

    train_paths, train_labels = get_paths_and_labels(train_dir, class_map)
    val_paths, val_labels = get_paths_and_labels(val_dir, class_map)

    temp_train_data = list(zip(train_paths, train_labels))
    random.shuffle(temp_train_data)
    train_paths, train_labels = zip(*temp_train_data)

    logging.info(f"Found {len(train_paths)} images for training and {len(val_paths)} for validation.")

    # Use the Albumentations-based dataset and transforms
    train_dataset = AlbumentationsDataset(train_paths, train_labels, transform=get_transforms(is_train=True))
    val_dataset = AlbumentationsDataset(val_paths, val_labels, transform=get_transforms(is_train=False))

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=2)
    logging.info("Data loaders created.")

    # --- Model, Optimizer, Loss ---
    model = AEyeModel(config['model_config']).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=0.01)
    
    # --- Standardized Weighted Loss Calculation ---
    num_mature = len(glob.glob(os.path.join(train_dir, 'mature', '*')))
    num_immature = len(glob.glob(os.path.join(train_dir, 'immature', '*')))
    weight = 1.0
    if num_immature > 0 and num_mature > 0:
        weight = num_immature / num_mature
    pos_weight = torch.tensor([weight], device=device)
    logging.info(f"Using weighted loss. Weight for 'mature' class: {weight:.2f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])

    logging.info("Model, optimizer, and loss function initialized.")
    logging.info(f"Model Configuration: {config}")

    best_val_f1 = 0.0
    os.makedirs(config['save_dir'], exist_ok=True)

    # --- Training & Validation Loops ---
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

    # Model configuration
    model_config = {
        'dims': [16, 32, 96, 128],
        'embed_dim': 192,
    }
    
    config = {'model_config': model_config}
    config.update(vars(args))

    train_model(config)