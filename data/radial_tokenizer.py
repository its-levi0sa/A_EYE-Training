import torch
import torch.nn as nn
import numpy as np

class RadialTokenizer(nn.Module):
    """An optimized, fully vectorized RadialTokenizer that runs on the GPU."""
    def __init__(self, image_size=256, num_rings=4):
        super().__init__()
        self.image_size = image_size
        self.center = (image_size // 2, image_size // 2)

        if num_rings == 4: ring_width = 32
        elif num_rings == 8: ring_width = 16
        elif num_rings == 16: ring_width = 8
        else: raise ValueError("Unsupported number of rings. Must be 4, 8, or 16.")
            
        self.rings = [(i * ring_width, (i + 1) * ring_width) for i in range(num_rings)]
        self.num_rings = len(self.rings)

        y, x = torch.meshgrid(torch.arange(0, image_size), torch.arange(0, image_size), indexing='ij')
        distance_grid = torch.sqrt((x - self.center[0])**2 + (y - self.center[1])**2)
        
        ring_masks = []
        for r_inner, r_outer in self.rings:
            mask = (distance_grid >= r_inner) & (distance_grid < r_outer)
            ring_masks.append(mask)
            
        self.register_buffer('ring_masks', torch.stack(ring_masks, dim=0).float())

    def forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
        B, C, H, W = image_tensor.shape
        device = image_tensor.device
        masks = self.ring_masks.to(device)
        
        masked_pixels = masks.unsqueeze(0).unsqueeze(2) * image_tensor.unsqueeze(1)
        
        num_pixels_per_ring = masks.sum(dim=[1, 2]) + 1e-6

        sum_vals = masked_pixels.sum(dim=[3, 4])
        mean_vals = sum_vals / num_pixels_per_ring.view(1, self.num_rings, 1)

        sum_sq_vals = (masked_pixels**2).sum(dim=[3, 4])
        mean_sq_vals = sum_sq_vals / num_pixels_per_ring.view(1, self.num_rings, 1)
        std_vals = torch.sqrt(torch.clamp(mean_sq_vals - mean_vals**2, min=0))

        flat_pixels = masked_pixels.view(B, self.num_rings, C, -1)
        flat_pixels[flat_pixels == 0] = float('nan')
        median_vals = torch.nanmedian(flat_pixels, dim=3).values

        tokens = torch.cat([mean_vals, std_vals, median_vals], dim=2)
        
        return tokens.to(device=device, dtype=torch.float32)