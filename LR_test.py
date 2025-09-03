import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
import os
import argparse
import logging
from tqdm import tqdm
import glob
import random
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from model.aeye_model import AEyeModel

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        bce_loss = self.bce_loss(inputs, targets)
        p_t = torch.exp(-bce_loss)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * (1 - p_t)**self.gamma * bce_loss
        if self.reduction == 'mean': return focal_loss.mean()
        elif self.reduction == 'sum': return focal_loss.sum()
        else: return focal_loss

def get_transforms(is_train=True):
    if is_train:
        return A.Compose([
            A.Resize(256, 256),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.75),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=30, p=0.75),
            A.Blur(blur_limit=3, p=0.2),
            A.GridDistortion(p=0.2),
            A.OpticalDistortion(distort_limit=0.2, shift_limit=0.2, p=0.2),
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.5),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(256, 256),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2(),
        ])

class AlbumentationsDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = np.array(image_paths)
        self.labels = np.array(labels)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        return image, label

def test_one_lr(train_loader, val_loader, config, learning_rate):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AEyeModel(config['model_config']).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=config['weight_decay'])
    criterion = FocalLoss()
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, steps_per_epoch=len(train_loader), epochs=config['epochs'])
    scaler = torch.cuda.amp.GradScaler()
    best_val_f1 = 0.0
    patience = 15
    epochs_no_improve = 0

    logging.info(f"--- Testing LR: {learning_rate} ---")

    for epoch in range(config['epochs']):
        model.train()
        train_loop = tqdm(train_loader, desc=f"LR Test, Epoch {epoch+1}/{config['epochs']}")
        for inputs, labels in train_loop:
            inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            train_loop.set_postfix(loss=loss.item())

        model.eval()
        val_preds, val_labels_all = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                preds = torch.sigmoid(outputs) > 0.5
                val_preds.extend(preds.cpu().numpy().flatten())
                val_labels_all.extend(labels.cpu().numpy().flatten())

        f1 = f1_score(val_labels_all, val_preds, zero_division=0)

        if f1 > best_val_f1:
            best_val_f1 = f1
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            logging.info(f"LR {learning_rate}: Early stopping at epoch {epoch+1}. Best F1: {best_val_f1:.4f}")
            break
            
    return best_val_f1

def main(config):
    seed_everything(seed=42)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    data_dir = 'data/train'
    class_map = {'immature': 0, 'mature': 1}

    def get_paths_and_labels(data_dir, class_mapping):
        all_paths, all_labels = [], []
        for class_name, label in class_mapping.items():
            class_path = os.path.join(data_dir, class_name)
            image_paths = glob.glob(os.path.join(class_path, '*.[jp][pn]g'))
            all_paths.extend(image_paths)
            all_labels.extend([label] * len(image_paths))
        return np.array(all_paths), np.array(all_labels)

    image_paths, labels = get_paths_and_labels(data_dir, class_map)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, val_idx = next(iter(skf.split(image_paths, labels)))

    train_paths, val_paths = image_paths[train_idx], image_paths[val_idx]
    train_labels, val_labels = labels[train_idx], labels[val_idx]
    
    train_dataset = AlbumentationsDataset(train_paths, train_labels, transform=get_transforms(is_train=True))
    val_dataset = AlbumentationsDataset(val_paths, val_labels, transform=get_transforms(is_train=False))
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=2)
    
    lr_candidates = [2e-5, 5e-5, 1e-4, 2e-4, 5e-4]
    lr_scores = {}

    for lr in lr_candidates:
        f1 = test_one_lr(train_loader, val_loader, config, lr)
        lr_scores[lr] = f1

    best_lr = max(lr_scores, key=lr_scores.get)
    logging.info("\n--- LR Tuning Complete ---")
    for lr, score in lr_scores.items():
        logging.info(f"LR: {lr}, F1-Score: {score:.4f}")
    logging.info(f"Best Learning Rate Found: {best_lr}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Find Best LR for 8-Ring A-EYE Model")
    parser.add_argument('--epochs', type=int, default=100, help='Max epochs for LR testing')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='Weight decay')
    args = parser.parse_args()

    model_config = {
        'dims': [32, 64, 128, 160],
        'embed_dim': 256,
    }

    config = {'model_config': model_config}
    config.update(vars(args))
    main(config)