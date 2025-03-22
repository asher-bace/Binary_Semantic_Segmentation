import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

# 收縮路徑 (Contracting Path) #
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Double Convolutions #
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1)
        self.batchNorm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)
        self.batchNorm2 = nn.BatchNorm2d(out_channels)

        # Down Sampling，特徵圖縮小一半 #
        self.maxpool = nn.MaxPool2d(2, stride = 2)

    def forward(self, x):
        # input -> conv1 -> batchNorm1 -> ReLU -> conv2 -> batchNorm2 -> ReLU -> MaxPool -> output#
        x = F.relu(self.batchNorm1(self.conv1(x)))
        x = F.relu(self.batchNorm2(self.conv2(x)))
        skip = x  # 保留原始未採樣的特徵，供 Decoder Block 使用 (Skip Connection) #
        x = self.maxpool(x)

        return x, skip

# 擴展路徑 (Expansive Path) #
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Up Sampling，讓特徵圖放大兩倍 #
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size = 2, stride = 2)

        # Double Convolutions #
        self.conv1 = nn.Conv2d(out_channels * 2, out_channels, kernel_size = 3, padding = 1)
        self.batchNorm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)
        self.batchNorm2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, skip):
        # input -> up conv -> concatenate skip -> conv1 -> batchNorm1 -> conv2 -> batchNorm2 -> ReLU -> output #
        x = self.upconv(x)
        x = torch.cat([x, skip], dim = 1)  # 將放大的特徵圖與未採樣的特徵圖進行接合 #
        x = F.relu(self.batchNorm1(self.conv1(x)))
        x = F.relu(self.batchNorm2(self.conv2(x)))

        return x

# U-Net 架構 #
class UNet(nn.Module):
    def __init__(self, input_channels = 3, num_classes = 2):
        super().__init__()

        # 收縮路徑 (Contracting Path) #
        self.encoder1 = EncoderBlock(input_channels, 64)
        self.encoder2 = EncoderBlock(64, 128)
        self.encoder3 = EncoderBlock(128, 256)
        self.encoder4 = EncoderBlock(256, 512)

        # Bottleneck #
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )

        # 擴展路徑 (Expansive Path) #
        self.decoder4 = DecoderBlock(1024, 512)
        self.decoder3 = DecoderBlock(512, 256)
        self.decoder2 = DecoderBlock(256, 128)
        self.decoder1 = DecoderBlock(128, 64)

        # 最終輸出層 (Final Output Layer) #
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # 收縮路徑 (Contracting Path) #
        x1, skip1 = self.encoder1(x)
        x2, skip2 = self.encoder2(x1)
        x3, skip3 = self.encoder3(x2)
        x4, skip4 = self.encoder4(x3)
        
        # Bottleneck #
        x = self.bottleneck(x4)
        
        # 擴展路徑 (Expansive Path) #
        x = self.decoder4(x, skip4)
        x = self.decoder3(x, skip3)
        x = self.decoder2(x, skip2)
        x = self.decoder1(x, skip1)
        
        # Final output
        x = self.final_conv(x)
        return x

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet(input_channels=3, num_classes=2).to(device)
    summary(model, input_size=(1, 3, 256, 256), device=device)
