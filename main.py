from encoder import Encoder
from siamese import SiameseNetwork
from dataset import dataloader
import torch.nn as nn
import torch.optim as optim
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

encoder = Encoder().to(device)
siamese_network = SiameseNetwork(encoder).to(device)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(siamese_network.parameters(), lr=1e-4)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    siamese_network.train()
    epoch_loss = 0.0

    for img1, img2, labels in dataloader:
        img1, img2, labels = img1.to(device), img2.to(device), labels.to(device).unsqueeze(1)

        # Forward pass
        optimizer.zero_grad()
        outputs = siamese_network(img1, img2)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(dataloader):.4f}")
