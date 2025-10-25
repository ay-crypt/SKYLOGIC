# models/eo_backbone.py

import torch
import torch.nn as nn
import torchvision.models as models

class EOEmbeddingModel(nn.Module):
    """
    A customizable embedding model for EO data.
    Uses a standard CNN backbone but adapts its first layer for multi-channel input.
    """
    def __init__(self, backbone_name, num_input_channels, embedding_dim, pretrained_imagenet=False):
        super().__init__()

        # Load the specified backbone
        if backbone_name == "resnet18":
            self.backbone = models.resnet18(pretrained=pretrained_imagenet)
        elif backbone_name == "resnet50":
            self.backbone = models.resnet50(pretrained=pretrained_imagenet)
        # Add other backbones as needed (EfficientNet, VGG, etc.)
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        # Adapt the first convolutional layer for multi-channel input
        # Get the original first conv layer
        original_conv1 = self.backbone.conv1
        
        # Create a new conv layer with the desired number of input channels
        # Keep other parameters (out_channels, kernel_size, stride, padding, bias) the same
        self.backbone.conv1 = nn.Conv2d(
            in_channels=num_input_channels,
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias
        )
        
        # If not loading ImageNet pretrained weights, you might want to initialize this new layer
        if not pretrained_imagenet:
            nn.init.kaiming_normal_(self.backbone.conv1.weight, mode='fan_out', nonlinearity='relu')
            if self.backbone.conv1.bias is not None:
                nn.init.constant_(self.backbone.conv1.bias, 0)
        else:
            # If pretrained_imagenet is True, we need to carefully handle the new conv1.
            # A common approach is to take the mean of the original 3 channels' weights
            # and replicate it across all new input channels. Or, initialize it randomly.
            # Here, we'll initialize randomly to avoid potential mismatch issues
            # given it's now a different number of channels. Fine-tuning will adjust it.
            print("WARNING: Using ImageNet pre-trained weights with adapted conv1. The new conv1 layer is randomly initialized.")
            nn.init.kaiming_normal_(self.backbone.conv1.weight, mode='fan_out', nonlinearity='relu')
            if self.backbone.conv1.bias is not None:
                nn.init.constant_(self.backbone.conv1.bias, 0)


        # Remove the original fully connected layer (classifier head)
        # We only want the features before classification
        self.backbone.fc = nn.Identity() # Replaces the FC layer with an identity module

        # Add a custom embedding head
        # The output of the backbone (before FC) for ResNet is typically 512 for ResNet18
        # or 2048 for ResNet50
        if backbone_name == "resnet18":
            backbone_output_dim = 512
        elif backbone_name == "resnet50":
            backbone_output_dim = 2048
        else:
            # Fallback or more robust way to get last layer's output dim
            # (e.g., by passing a dummy tensor through)
            backbone_output_dim = self._get_backbone_output_dim(num_input_channels, backbone_name)

        self.embedding_head = nn.Sequential(
            nn.Linear(backbone_output_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True)
            # You might add more layers here, e.g., another linear layer or a dropout
        )

    def _get_backbone_output_dim(self, num_input_channels, backbone_name):
        """Helper to dynamically get the output dimension of the backbone before the FC layer."""
        with torch.no_grad():
            dummy_input = torch.randn(1, num_input_channels, 224, 224) # Standard input size for ResNet
            # Create a temporary backbone instance to get its output shape
            if backbone_name == "resnet18":
                temp_backbone = models.resnet18(pretrained=False)
            elif backbone_name == "resnet50":
                temp_backbone = models.resnet50(pretrained=False)
            else:
                raise ValueError(f"Unsupported backbone for dynamic dim check: {backbone_name}")

            temp_backbone.conv1 = nn.Conv2d(
                in_channels=num_input_channels,
                out_channels=temp_backbone.conv1.out_channels,
                kernel_size=temp_backbone.conv1.kernel_size,
                stride=temp_backbone.conv1.stride,
                padding=temp_backbone.conv1.padding,
                bias=temp_backbone.conv1.bias
            )
            temp_backbone.fc = nn.Identity()
            output = temp_backbone(dummy_input)
            return output.shape[1]


    def forward(self, x):
        features = self.backbone(x)
        # Flatten if necessary (backbone might output (Batch, Features, 1, 1) or similar)
        features = features.view(features.size(0), -1)
        embedding = self.embedding_head(features)
        # Normalize embeddings to unit hypersphere for better performance with distance-based losses
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        return embedding
