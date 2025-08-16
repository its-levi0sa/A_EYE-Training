from torch.utils.data import Dataset
from PIL import Image
import torch

class CataractDataset(Dataset):
    """Loads preprocessed images and labels from file paths."""
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            print(f"Warning: File not found at {image_path}. Returning a dummy image.")
            image = Image.new('RGB', (128, 128), color = 'red')

        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, label