# config.py

import os

# --- General Configuration ---
PROJECT_NAME = "eo-embedding-training"
DEVICE = "cuda" # or "cpu"
SEED = 42

# --- Data Configuration ---
# Base directory where your EO GeoTIFFs are stored
# For demonstration, let's assume images are in a structure like:
# DATA_DIR/
# ├── agriculture/
# │   ├── scene_1.tif
# │   ├── scene_2.tif
# ├── forest/
# │   ├── scene_a.tif
# │   ├── scene_b.tif
# └── urban/
#     ├── patch_x.tif
#     └── patch_y.tif
DATA_DIR = os.path.join(os.getcwd(), "eo_data_samples") # Make sure this path exists and contains data

# Specify the desired bands to load from your GeoTIFFs
# IMPORTANT: These indices are 0-based based on the order in your GeoTIFF.
# For Sentinel-2, a common order for bands B2,B3,B4,B8,B11 might be 0,1,2,3,4
# You need to know the band order of YOUR specific GeoTIFFs.
# Example for a hypothetical 5-band image: Blue, Green, Red, NIR, SWIR1
BAND_INDICES_TO_LOAD = [0, 1, 2, 3, 4] # Example: B, G, R, NIR, SWIR1
NUM_INPUT_BANDS = len(BAND_INDICES_TO_LOAD)

# List of spectral indices to calculate and add as additional channels
# Ensure the bands required for these indices are in BAND_INDICES_TO_LOAD
CALCULATE_INDICES = ["NDVI", "NDWI"] # Example: Normalized Difference Vegetation Index, Water Index

# Spatial dimensions of input patches (e.g., 64x64, 128x128)
PATCH_SIZE = (128, 128)

# --- Model Configuration ---
BACKBONE_NAME = "resnet18" # Or "resnet50", etc.
PRETRAINED_IMAGENET = False # Typically False for EO, but can be a starting point
EMBEDDING_DIM = 128 # Output dimension of the embedding vector

# --- Training Configuration ---
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

# Triplet Loss parameters
TRIPLET_MARGIN = 0.2
