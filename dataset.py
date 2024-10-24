import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import cv2

class PostureDataset(Dataset):
    def __init__(self, img1_paths, img2_paths, labels, transform=None):
        self.img1_paths = img1_paths
        self.img2_paths = img2_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Load and preprocess images
        img1 = cv2.imread(self.img1_paths[idx])
        img2 = cv2.imread(self.img2_paths[idx])

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return img1, img2, label

# Example: Dummy paths and labels (replace with actual data)
img1_paths = ["test.jpeg"] * 100
img2_paths = ["test2.png"] * 100
labels = np.random.rand(100)  # Continuous labels

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = PostureDataset(img1_paths, img2_paths, labels, transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
