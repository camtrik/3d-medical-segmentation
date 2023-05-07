import torch 
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np

import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super().__init__()
        # when down: mid = in_channels, when up: mid = out_channels
        if not mid_channels:
            mid_channels = out_channels if in_channels > out_channels else in_channels

        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, down=True):
        super().__init__()
        self.down = nn.MaxPool3d(2) if down else None
        self.conv = DoubleConv(in_channels, out_channels, mid_channels)

    def forward(self, x):
        x = self.down(x) if self.down else x
        return self.conv(x)

    
class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, in_channels, kernel_size=2, stride=2)
    
        self.conv = DoubleConv(in_channels + in_channels // 2, out_channels, mid_channels)
    
    def forward(self, x, encoder_x):
        x = self.up(x)

        # diffZ = x.size()[2] - encoder_x.size()[2]
        # diffX = x.size()[3] - encoder_x.size()[3]
        # diffY = x.size()[4] - encoder_x.size()[4]

        x = torch.cat([x, encoder_x], dim=1)
        return self.conv(x)
    
class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.encoder1 = Encoder(in_channels, 64, mid_channels=32, down=False)
        self.encoder2 = Encoder(64, 128)
        self.encoder3 = Encoder(128, 256)
        self.encoder4 = Encoder(256, 512)

        self.decoder1 = Decoder(512, 256)
        self.decoder2 = Decoder(256, 128)
        self.decoder3 = Decoder(128, 64)

        self.out = nn.Conv3d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.encoder1(x) # 64
        x2 = self.encoder2(x1) # 128
        x3 = self.encoder3(x2) # 256
        x = self.encoder4(x3) # 512

        x = self.decoder1(x, x3) # 512 + 256
        x = self.decoder2(x, x2) # 256 + 128
        x = self.decoder3(x, x1) # 128 + 64

        return self.out(x)
