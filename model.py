import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels=5, out_channels=1):
        super(UNet, self).__init__()

        # Encoder (left side)
        self.enc1 = self.conv_block(in_channels, 32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)  # 160×224 -> 80×112
        self.enc2 = self.conv_block(32, 64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)  # 80×112 -> 40×56
        self.enc3 = self.conv_block(64, 128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)  # 40×56 -> 20×28
        self.enc4 = self.conv_block(128, 256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)  # 20×28 -> 10×14
        self.enc5 = self.conv_block(256, 512)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)  # 10×14 -> 5×7

        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)

        # Decoder (right side)
        self.upconv5 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)  # 5×7 -> 10×14
        self.dec5 = self.conv_block(1024, 512)
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)  # 10×14 -> 20×28
        self.dec4 = self.conv_block(512, 256)
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)  # 20×28 -> 40×56
        self.dec3 = self.conv_block(256, 128)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)  # 40×56 -> 80×112
        self.dec2 = self.conv_block(128, 64)
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)  # 80×112 -> 160×224
        self.dec1 = self.conv_block(64, 32)

        # Output layer
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def crop(self, enc_feature, dec_feature):
        """Crop encoder feature map to match decoder feature map size"""
        _, _, h, w = dec_feature.size()
        return enc_feature[:, :, :h, :w]

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)  # [Batch, 32, 160, 224]
        enc2 = self.enc2(self.pool1(enc1))  # [Batch, 64, 80, 112]
        enc3 = self.enc3(self.pool2(enc2))  # [Batch, 128, 40, 56]
        enc4 = self.enc4(self.pool3(enc3))  # [Batch, 256, 20, 28]
        enc5 = self.enc5(self.pool4(enc4))  # [Batch, 512, 10, 14]

        # Bottleneck
        bottleneck = self.bottleneck(self.pool5(enc5))  # [Batch, 1024, 5, 7]

        # Decoder
        dec5 = self.dec5(torch.cat([self.upconv5(bottleneck), self.crop(enc5, self.upconv5(bottleneck))], dim=1))  # [Batch, 512, 10, 14]
        dec4 = self.dec4(torch.cat([self.upconv4(dec5), self.crop(enc4, self.upconv4(dec5))], dim=1))              # [Batch, 256, 20, 28]
        dec3 = self.dec3(torch.cat([self.upconv3(dec4), self.crop(enc3, self.upconv3(dec4))], dim=1))              # [Batch, 128, 40, 56]
        dec2 = self.dec2(torch.cat([self.upconv2(dec3), self.crop(enc2, self.upconv2(dec3))], dim=1))              # [Batch, 64, 80, 112]
        dec1 = self.dec1(torch.cat([self.upconv1(dec2), self.crop(enc1, self.upconv1(dec2))], dim=1))              # [Batch, 32, 160, 224]

        # Output
        return self.final_conv(dec1)  # [Batch, 1, 160, 224]
