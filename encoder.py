import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_shape=(224,224,3), latent_dim=64):
        super(Encoder, self).__init__()
        self.input_shape =  input_shape
        self.latent_dim = latent_dim

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1),
            nn.LeakyReLU(),
            nn.Flatten()
        )
        self.fc = nn.Linear(128*self.input_shape[0]**2, self.latent_dim)

    def forward(self, x):
        x = self.cnn(x)
        x = self.fc(x)
        return x

