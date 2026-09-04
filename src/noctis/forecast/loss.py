import torch.nn as nn

class PersistenceDefeatingLoss(nn.Module):
    def __init__(self, beta=0.1):
        super().__init__()
        self.smooth_l1 = nn.SmoothL1Loss(beta=beta)
    def forward(self, pred, target):
        return self.smooth_l1(pred, target)