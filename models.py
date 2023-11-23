import torch

class SegModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # encoder
        self.encoder = torch.nn.Sequential(  # 256
            torch.nn.Conv2d(3, 32, 3, 2, 1),  # 128
            torch.nn.Conv2d(32, 64, 3, 2, 1),  # 64
            torch.nn.Conv2d(64, 128, 3, 2, 1),  # 32
            torch.nn.Conv2d(128, 256, 3, 2, 1),  # 16
        )
        # decoder
        self.decoder = torch.nn.Sequential(  # 16
            torch.nn.ConvTranspose2d(256, 128, 3, 2, 1),  # 32
            torch.nn.ConvTranspose2d(128, 64, 3, 2, 1),  # 64
            torch.nn.ConvTranspose2d(64, 32, 3, 2, 1),  # 128
            torch.nn.ConvTranspose2d(32, ds.num_classes, 3, 2, 1),  # 256
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x