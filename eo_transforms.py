# data/eo_transforms.py

import torch
import numpy as np
import torchvision.transforms as T
import math

class NormalizeEO:
    """
    Normalize EO data.
    Assumes input tensor is (C, H, W).
    Can normalize based on global min/max or z-score.
    For reflectance data, often a simple 0-1 scale is sufficient.
    """
    def __init__(self, method="min_max", min_val=0, max_val=10000, mean=None, std=None):
        self.method = method
        self.min_val = min_val
        self.max_val = max_val
        self.mean = mean
        self.std = std

    def __call__(self, img_tensor):
        if self.method == "min_max":
            # Clip and scale to 0-1
            img_tensor = torch.clamp(img_tensor, self.min_val, self.max_val)
            img_tensor = (img_tensor - self.min_val) / (self.max_val - self.min_val)
        elif self.method == "z_score":
            if self.mean is None or self.std is None:
                raise ValueError("Mean and Std must be provided for z-score normalization.")
            img_tensor = (img_tensor - self.mean.to(img_tensor.device)) / (self.std.to(img_tensor.device) + 1e-6)
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")
        return img_tensor

class CalculateSpectralIndices:
    """
    Calculates specified spectral indices and appends them as new channels.
    Assumes input tensor is (C, H, W) and band order is known (defined in config).
    Requires a mapping of band names to channel indices.
    """
    def __init__(self, indices_to_calculate, band_names_to_idx_map):
        self.indices_to_calculate = indices_to_calculate
        self.band_names_to_idx_map = band_names_to_idx_map

    def _safe_divide(self, numerator, denominator):
        """Avoids division by zero, replaces NaNs with 0."""
        # Add a small epsilon to denominator to prevent division by zero
        denominator = torch.where(denominator == 0, torch.tensor(1e-6, device=denominator.device), denominator)
        result = numerator / denominator
        # Replace any potential remaining NaNs (e.g., if both numerator and denominator were 0 and epsilon was not enough)
        result = torch.nan_to_num(result, nan=0.0)
        return result

    def _calculate_ndvi(self, img_tensor):
        red_idx = self.band_names_to_idx_map.get("Red")
        nir_idx = self.band_names_to_idx_map.get("NIR")
        if red_idx is None or nir_idx is None:
            print("Warning: Red or NIR band not found for NDVI calculation.")
            return None
        red = img_tensor[red_idx]
        nir = img_tensor[nir_idx]
        return self._safe_divide(nir - red, nir + red)

    def _calculate_ndwi(self, img_tensor):
        green_idx = self.band_names_to_idx_map.get("Green")
        nir_idx = self.band_names_to_idx_map.get("NIR")
        if green_idx is None or nir_idx is None:
            print("Warning: Green or NIR band not found for NDWI calculation.")
            return None
        green = img_tensor[green_idx]
        nir = img_tensor[nir_idx]
        return self._safe_divide(green - nir, green + nir)
    
    # Add more index calculations as needed (e.g., EVI, NBR, etc.)

    def __call__(self, img_tensor):
        calculated_indices = []
        if "NDVI" in self.indices_to_calculate:
            ndvi = self._calculate_ndvi(img_tensor)
            if ndvi is not None:
                calculated_indices.append(ndvi.unsqueeze(0))
        if "NDWI" in self.indices_to_calculate:
            ndwi = self._calculate_ndwi(img_tensor)
            if ndwi is not None:
                calculated_indices.append(ndwi.unsqueeze(0))
        
        if calculated_indices:
            # Stack new indices as new channels
            return torch.cat([img_tensor] + calculated_indices, dim=0)
        return img_tensor


def get_eo_transforms(patch_size, band_names_to_idx_map, indices_to_calculate, is_train=True):
    """
    Defines the set of transformations for EO data.
    """
    transform_list = []

    # Calculate indices first, as other transforms might benefit
    if indices_to_calculate:
        transform_list.append(CalculateSpectralIndices(indices_to_calculate, band_names_to_idx_map))

    # Resizing / Cropping
    # Assuming the input GeoTIFFs might not be exactly PATCH_SIZE
    # You might want to Pad if smaller, or RandomCrop if larger
    # For simplicity, let's just resize for now. A RandomCrop would be better for training.
    # Note: Interpolation for multi-band data should ideally be nearest neighbor or bilinear
    # if dealing with discrete data (like masks) or continuous data respectively.
    # Here, we treat all bands equally for resizing.
    transform_list.append(T.Resize(patch_size, interpolation=T.InterpolationMode.BILINEAR))


    if is_train:
        # EO-specific augmentations
        # Apply random horizontal/vertical flips. Ensure consistency across bands.
        transform_list.append(T.RandomHorizontalFlip())
        transform_list.append(T.RandomVerticalFlip())
        # Random rotation
        transform_list.append(T.RandomRotation(degrees=90)) # Rotate by 0, 90, 180, 270 degrees

        # Other augmentations like color jitter are less applicable to raw EO bands.
        # However, you could consider random brightness/contrast if it's applied consistently across bands
        # and makes sense for the specific data (e.g., if you're dealing with varying illumination).
        # For true spectral bands, this needs careful consideration.

    # Normalize after all augmentations
    # Example: Sentinel-2 reflectance values are often 0-10000. Normalize to 0-1.
    # If using z-score, you'd calculate mean/std over your entire dataset.
    transform_list.append(NormalizeEO(method="min_max", min_val=0, max_val=10000))

    return T.Compose(transform_list)
