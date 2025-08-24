import torch
import torch.nn as nn
import cv2
import numpy as np

class RadialTokenizer(nn.Module):
    """
    Custom RadialTokenizer from file.
    """
    def __init__(self):
        super().__init__()
        self.center = (128, 128)
        self.rings = [(i * 8, (i + 1) * 8) for i in range(16)]

    def _create_ring_mask(self, shape, center, inner_r, outer_r):
        mask = np.zeros(shape[:2], dtype=np.uint8)
        cv2.circle(mask, center, outer_r, 255, -1)
        cv2.circle(mask, center, inner_r, 0, -1)
        return mask

    def _extract_ring_features(self, image, mask):
        masked = cv2.bitwise_and(image, image, mask=mask)
        pixels = masked[mask == 255]
        if pixels.shape[0] == 0:
            return np.zeros(9)
        mean = pixels.mean(axis=0)
        std = pixels.std(axis=0)
        median = np.median(pixels, axis=0)
        return np.concatenate([mean, std, median])

    def forward(self, image_tensor):
        B = image_tensor.shape[0]
        device = image_tensor.device
        tokens_9d_list = []
        for img in image_tensor:
            # Denormalize and convert to numpy for OpenCV processing
            img_np = img.permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * 0.5) + 0.5 # Assuming normalization was mean=0.5, std=0.5
            img_np = (img_np * 255.0).astype(np.uint8)

            ring_features = []
            for r0, r1 in self.rings:
                mask = self._create_ring_mask(img_np.shape, self.center, r0, r1)
                ring_feat = self._extract_ring_features(img_np, mask)
                ring_features.append(ring_feat)
            tokens_9d_list.append(ring_features)

        tokens_9d = torch.from_numpy(np.array(tokens_9d_list)).to(device=device, dtype=torch.float32)
        return tokens_9d