import torch.nn as nn
import torch

class SiameseNetwork(nn.Module):
    def __init__(self, encoder):
        super(SiameseNetwork, self).__init__()
        self.encoder = encoder
        self.fc = nn.Linear(1, 1)  # Predicts a continuous similarity score

    def forward(self, img1, img2):
        # Get embeddings from the shared encoder
        embed1 = self.encoder(img1)
        embed2 = self.encoder(img2)

        # Compute L2 distance
        l2_distance = torch.sqrt(torch.sum((embed1 - embed2) ** 2, dim=1, keepdim=True))
        
        # Predict similarity score
        score = self.fc(l2_distance)
        return score