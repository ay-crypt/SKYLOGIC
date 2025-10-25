# losses/triplet_loss.py

import torch.nn as nn

class TripletLoss(nn.TripletMarginLoss):
    """
    Standard Triplet Margin Loss wrapper.
    """
    def __init__(self, margin=1.0, p=2, eps=1e-6, swap=False):
        super().__init__(margin=margin, p=p, eps=eps, swap=swap)
