# train.py

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import random
import numpy as np

# Import components from our new EO setup
import config
from data.eo_transforms import get_eo_transforms
from data.EO_Dataset import EODataset
from models.eo_backbone import EOEmbeddingModel
from losses.triplet_loss import TripletLoss # Or your custom loss

def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    set_seed(config.SEED)

    # --- 1. Prepare Data ---
    # Define a simple mapping for common bands to indices.
    # THIS MUST MATCH THE ORDER OF BANDS IN YOUR GEOTIFFS AND config.BAND_INDICES_TO_LOAD
    # Example: if BAND_INDICES_TO_LOAD = [0,1,2,3,4] and these correspond to
    # Blue, Green, Red, NIR, SWIR1, then:
    band_names_to_idx_map = {
        "Blue": 0, "Green": 1, "Red": 2, "NIR": 3, "SWIR1": 4
    }
    
    # Ensure the config NUM_INPUT_BANDS includes potential indices
    # We will adjust config.NUM_INPUT_BANDS after `get_eo_transforms`
    initial_num_bands = config.NUM_INPUT_BANDS
    
    # Get transformations
    train_transforms = get_eo_transforms(
        patch_size=config.PATCH_SIZE,
        band_names_to_idx_map=band_names_to_idx_map,
        indices_to_calculate=config.CALCULATE_INDICES,
        is_train=True
    )
    val_transforms = get_eo_transforms(
        patch_size=config.PATCH_SIZE,
        band_names_to_idx_map=band_names_to_idx_map,
        indices_to_calculate=config.CALCULATE_INDICES,
        is_train=False
    )

    # Create dataset instances
    try:
        train_dataset = EODataset(
            data_dir=config.DATA_DIR,
            band_indices_to_load=config.BAND_INDICES_TO_LOAD,
            eo_transforms=train_transforms,
            mode='train'
        )
        # Update NUM_INPUT_BANDS based on how many channels the dataset outputs after transforms
        # This is a bit of a hack; ideally, you'd calculate this during dataset init.
        # For a more robust approach, pass a dummy tensor through transforms to get final shape.
        dummy_tensor = train_dataset[0][0] # Get one anchor image to infer final channel count
        config.NUM_INPUT_BANDS = dummy_tensor.shape[0] # Update global config
        print(f"Final number of input channels for model: {config.NUM_INPUT_BANDS} (original bands + calculated indices)")

        val_dataset = EODataset(
            data_dir=config.DATA_DIR, # For simplicity, using same data for val, but split properly in real projects
            band_indices_to_load=config.BAND_INDICES_TO_LOAD,
            eo_transforms=val_transforms,
            mode='val'
        )
    except RuntimeError as e:
        print(f"Error initializing datasets: {e}")
        print("Please ensure 'eo_data_samples' directory exists and contains GeoTIFFs categorized by class.")
        print("Example structure: eo_data_samples/forest/scene1.tif, eo_data_samples/urban/scene2.tif")
        return


    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=os.cpu_count(), # Use all available CPU cores
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=os.cpu_count(),
        pin_memory=True
    )

    # --- 2. Initialize Model, Loss, Optimizer ---
    model = EOEmbeddingModel(
        backbone_name=config.BACKBONE_NAME,
        num_input_channels=config.NUM_INPUT_BANDS,
        embedding_dim=config.EMBEDDING_DIM,
        pretrained_imagenet=config.PRETRAINED_IMAGENET
    ).to(config.DEVICE)

    criterion = TripletLoss(margin=config.TRIPLET_MARGIN).to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    # --- 3. Training Loop ---
    best_loss = float('inf')

    for epoch in range(config.NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} (Train)")

        for anchor, positive, negative in train_pbar:
            anchor = anchor.to(config.DEVICE)
            positive = positive.to(config.DEVICE)
            negative = negative.to(config.DEVICE)

            optimizer.zero_grad()

            anchor_embedding = model(anchor)
            positive_embedding = model(positive)
            negative_embedding = model(negative)

            loss = criterion(anchor_embedding, positive_embedding, negative_embedding)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_pbar.set_postfix(loss=loss.item())

        epoch_train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1} Train Loss: {epoch_train_loss:.4f}")

        # --- 4. Validation Loop ---
        model.eval()
        val_loss = 0.0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} (Validation)")

        with torch.no_grad():
            for anchor, positive, negative in val_pbar:
                anchor = anchor.to(config.DEVICE)
                positive = positive.to(config.DEVICE)
                negative = negative.to(config.DEVICE)

                anchor_embedding = model(anchor)
                positive_embedding = model(positive)
                negative_embedding = model(negative)

                loss = criterion(anchor_embedding, positive_embedding, negative_embedding)
                val_loss += loss.item()
                val_pbar.set_postfix(loss=loss.item())

        epoch_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1} Validation Loss: {epoch_val_loss:.4f}")

        # Save best model
        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), os.path.join("checkpoints", f"{config.PROJECT_NAME}_best_model.pth"))
            print(f"Saved best model with validation loss: {best_loss:.4f}")

    print("Training complete.")

if __name__ == "__main__":
    # Create a dummy data directory for demonstration
    # In a real scenario, you'd populate config.DATA_DIR with your actual data
    dummy_data_path = config.DATA_DIR
    if not os.path.exists(dummy_data_path):
        print(f"Creating dummy data directory at {dummy_data_path}")
        os.makedirs(os.path.join(dummy_data_path, "forest"), exist_ok=True)
        os.makedirs(os.path.join(dummy_data_path, "urban"), exist_ok=True)
        # Placeholder for creating dummy GeoTIFFs if needed for first run testing
        # This requires more complex rasterio code, so for now, manually place some .tif files.
        print("Please populate 'eo_data_samples/forest/' and 'eo_data_samples/urban/' with actual multi-band GeoTIFF files.")
        print("You can use tools like QGIS or GDAL to create small sample GeoTIFFs with 5 bands (e.g., from Sentinel-2 data).")
        print("Skipping execution as dummy data is not present.")
    else:
        main()
