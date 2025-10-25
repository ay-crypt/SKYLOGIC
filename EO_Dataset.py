# data/EO_Dataset.py

import torch
import numpy as np
import rasterio
from torch.utils.data import Dataset
from glob import glob
import os
import random

class EODataset(Dataset):
    """
    A PyTorch Dataset for loading Earth Observation (EO) data from local GeoTIFF files.
    Supports multi-band images and provides anchors, positives, and negatives for triplet loss.
    """
    def __init__(self, data_dir, band_indices_to_load, eo_transforms, mode='train'):
        self.data_dir = data_dir
        self.band_indices_to_load = band_indices_to_load
        self.eo_transforms = eo_transforms
        self.mode = mode

        # Store a mapping of band names to their 0-based index in the loaded tensor
        # This is crucial for spectral index calculation.
        # For simplicity, let's hardcode some common band names for now.
        # In a real scenario, you'd extract this from GeoTIFF metadata or config.
        # Assuming BAND_INDICES_TO_LOAD corresponds to (Blue, Green, Red, NIR, SWIR1) for example
        self.band_names_map = {
            "Blue": 0, "Green": 1, "Red": 2, "NIR": 3, "SWIR1": 4
        }
        
        # Discover all GeoTIFF files and categorize them by a pseudo-class (e.g., folder name)
        # This is a simple way to create groups for triplet mining
        self.image_paths_by_class = self._discover_files_and_classes()
        self.all_image_paths = [path for class_list in self.image_paths_by_class.values() for path in class_list]
        
        # Create a list of (image_path, class_label) tuples for simpler indexing
        self.indexed_data = []
        for class_name, paths in self.image_paths_by_class.items():
            for path in paths:
                self.indexed_data.append((path, class_name))

        if not self.indexed_data:
            raise RuntimeError(f"No EO data found in {data_dir}. Please check your data directory and file structure.")

        print(f"Found {len(self.all_image_paths)} EO scenes categorized into {len(self.image_paths_by_class)} classes.")

    def _discover_files_and_classes(self):
        """
        Discovers GeoTIFF files and categorizes them by their parent directory name.
        Assumes a structure like: data_dir/class_name/scene.tif
        """
        image_paths_by_class = {}
        class_dirs = [d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))]

        for class_name in class_dirs:
            class_path = os.path.join(self.data_dir, class_name)
            tiffs_in_class = glob(os.path.join(class_path, "*.tif"))
            if tiffs_in_class:
                image_paths_by_class[class_name] = tiffs_in_class
        return image_paths_by_class

    def _load_and_preprocess_eo_scene(self, file_path):
        """Loads a multi-band GeoTIFF and applies transformations."""
        with rasterio.open(file_path) as src:
            # Read selected bands. Rasterio reads bands as (bands, height, width).
            # Convert to float32 for model input
            eo_data = src.read(self.band_indices_to_load + [1]) # +[1] is a hack to make 1-based indexing
            eo_data = eo_data.astype(np.float32)

            # Important: Rasterio uses 1-based indexing for .read() when passing a list,
            # but we want 0-based for internal processing, so adjust
            # The [1] above assumes our list is for `bands` parameter
            # If band_indices_to_load are 0-based, then it should be:
            # selected_bands = [idx + 1 for idx in self.band_indices_to_load]
            # eo_data = src.read(selected_bands)
            # For simplicity, assuming the provided BAND_INDICES_TO_LOAD are 0-based,
            # and we manually map them if src.read expects 1-based.
            # A more robust solution would involve checking src.indexes or creating a VRT.
            
            # For now, let's assume `src.read()` handles a list of 0-based indices if you pass `out_dtype`
            # or we manually stack. The simplest way to load *all* bands and then select
            # is to read everything and then index.

            full_data = src.read().astype(np.float32)
            # Select specific bands
            eo_data = full_data[self.band_indices_to_load, :, :]


        # Convert to PyTorch tensor
        img_tensor = torch.from_numpy(eo_data)

        # Apply transformations
        img_tensor = self.eo_transforms(img_tensor)
        return img_tensor

    def __len__(self):
        return len(self.indexed_data)

    def __getitem__(self, idx):
        # Anchor: The current image
        anchor_path, anchor_class = self.indexed_data[idx]
        anchor_img = self._load_and_preprocess_eo_scene(anchor_path)

        # Positive: Another image from the same class as the anchor
        positive_path = anchor_path
        while positive_path == anchor_path: # Ensure it's not the anchor itself
            positive_path = random.choice(self.image_paths_by_class[anchor_class])
        positive_img = self._load_and_preprocess_eo_scene(positive_path)

        # Negative: An image from a different class than the anchor
        negative_class = anchor_class
        while negative_class == anchor_class:
            negative_class = random.choice(list(self.image_paths_by_class.keys()))
        negative_path = random.choice(self.image_paths_by_class[negative_class])
        negative_img = self._load_and_preprocess_eo_scene(negative_path)

        return anchor_img, positive_img, negative_img
