import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()

        self.enc1 = self.double_conv(in_channels, base_channels, dropout_p=0.1)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = self.double_conv(base_channels, base_channels * 2, dropout_p=0.1)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = self.double_conv(base_channels * 2, base_channels * 4, dropout_p=0.1)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = self.double_conv(base_channels * 4, base_channels * 8, dropout_p=0.1)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = self.double_conv(base_channels * 8, base_channels * 16, dropout_p=0.1)

        self.up4  = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, 2, 2)
        self.dec4 = self.double_conv(base_channels * 16, base_channels * 8, dropout_p=0.1)

        self.up3  = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, 2)
        self.dec3 = self.double_conv(base_channels * 8, base_channels * 4, dropout_p=0.1)

        self.up2  = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, 2)
        self.dec2 = self.double_conv(base_channels * 4, base_channels * 2, dropout_p=0.1)

        self.up1  = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, 2)
        self.dec1 = self.double_conv(base_channels * 2, base_channels, dropout_p=0.1)

        self.final = nn.Conv2d(base_channels, 1, kernel_size=1)

    def double_conv(self, in_c, out_c, dropout_p=0.1):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),

            nn.Dropout2d(p=dropout_p),
        )

    def forward(self, x):

        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        e4 = self.enc4(p3)
        p4 = self.pool4(e4)

        b = self.bottleneck(p4)

        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return torch.sigmoid(self.final(d1))
