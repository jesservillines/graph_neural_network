import torch

class WeightedMSELoss(torch.nn.Module):
    def __init__(self, threshold=7):
        super(WeightedMSELoss, self).__init__()
        self.threshold = threshold
        
    def forward(self, pred, target):
        weights = torch.where(target > self.threshold,
                            torch.ones_like(target) * 2.0,
                            torch.ones_like(target))
        return torch.mean(weights * (pred - target) ** 2)