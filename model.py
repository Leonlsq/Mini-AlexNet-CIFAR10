import torch
import torch.nn as nn


class MiniAlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(

            # Conv1: RGB 32 * 32
            # 64 Channels Output
            nn.Conv2d(in_channels=3, out_channels=64,
                      kernel_size=3, stride=1, padding=1)


        )
