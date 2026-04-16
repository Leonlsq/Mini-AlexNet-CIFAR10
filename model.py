import torch
import torch.nn as nn


class MiniAlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            # Conv1: RGB 32 * 32
            # 64 Channels Output
            nn.Conv2d(in_channels=3, out_channels=64,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # 32 * 32 -> 16 * 16
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv2
            nn.Conv2d(in_channels=64, out_channels=192,
                      kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # 16x16 -> 8x8
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv3
            nn.Conv2d(384, 192, 3, 1),
            nn.ReLU(True),

            # Con4
            nn.Conv2d(384, 256, 3, 1),
            nn.ReLU(True),

            # Conv5
            nn.Conv2d(256, 256, 3, 1),
            nn.ReLU(True),

            # 8*8 -> 4*4
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
