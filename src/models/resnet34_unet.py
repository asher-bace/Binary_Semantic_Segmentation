import torch
import torch.nn as nn
from torchinfo import summary

# Encoder: ResNet34 #
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super().__init__()

        # Double Convolutions #
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride,
                               padding = 1, bias = False)
        self.batchNorm1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1,
                               padding = 1, bias = False)
        self.batchNorm2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.batchNorm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.batchNorm2(out)
        out += identity
        out = self.relu(out)
        return out
    
class ResNet34(nn.Module):
    def __init__(self):
        super().__init__()

        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.batchNorm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        
        self.layer1 = self._make_layer(64, 3)
        self.layer2 = self._make_layer(128, 4, stride = 2)
        self.layer3 = self._make_layer(256, 6, stride = 2)
        self.layer4 = self._make_layer(512, 3, stride = 2)

    def _make_layer(self, out_channels, blocks, stride = 1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size = 1, stride = stride,
                          bias = False),
                nn.BatchNorm2d(out_channels)
            )
        layers = []
        layers.append(ResNetBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(ResNetBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.batchNorm1(x)
        s0 = self.relu(x) 
        e1 = self.maxpool(s0)
        e2 = self.layer1(e1)
        e3 = self.layer2(e2)
        e4 = self.layer3(e3)
        e5 = self.layer4(e4)
        return s0, e2, e3, e4, e5

# Decoder: U-Net #
class DecoderBlock(nn.Module):
    def __init__(self, conv_in_channels, conv_out_channels, 
                 up_in_channels = None, up_out_channels = None):
        
        super().__init__()

        if up_in_channels is None:
            up_in_channels = conv_in_channels

        if up_out_channels is None:
            up_out_channels = conv_out_channels

        self.up = nn.ConvTranspose2d(up_in_channels, up_out_channels,
                                     kernel_size = 2, stride = 2)
        self.conv = nn.Sequential(
            nn.Conv2d(conv_in_channels, conv_out_channels, kernel_size = 3,
                      padding = 1, bias = False),
            nn.BatchNorm2d(conv_out_channels),
            nn.ReLU(inplace = True),
            nn.Conv2d(conv_out_channels, conv_out_channels, kernel_size = 3,
                      padding = 1, bias = False),
            nn.BatchNorm2d(conv_out_channels),
            nn.ReLU(inplace = True)
        )
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)
    
class UNetDecoder(nn.Module):
    def __init__(self, num_classes=2, filters=[64, 128, 256, 512]):
        super().__init__()
        # Bridge 處理 encoder 最深層的特徵
        self.bridge = nn.Sequential(
            nn.Conv2d(filters[3], filters[3] * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(filters[3] * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Decoder Blocks
        self.decoder1 = DecoderBlock(conv_in_channels=filters[3] * 2, conv_out_channels=filters[3])
        self.decoder2 = DecoderBlock(conv_in_channels=filters[3], conv_out_channels=filters[2])
        self.decoder3 = DecoderBlock(conv_in_channels=filters[2], conv_out_channels=filters[1])
        self.decoder4 = DecoderBlock(conv_in_channels=filters[1], conv_out_channels=filters[0])
        # 最後一層：將 decoder4 上採樣後的特徵與 s0 (conv1 後的高解析度特徵)做 concat
        self.decoder5 = DecoderBlock(conv_in_channels=filters[0] + filters[0], 
                                     conv_out_channels=filters[0],
                                     up_in_channels=filters[0],
                                     up_out_channels=filters[0])
        self.lastlayer = nn.Sequential(
            nn.ConvTranspose2d(in_channels=filters[0], out_channels=filters[0],
                               kernel_size=2, stride=2),
            nn.Conv2d(filters[0], num_classes, kernel_size=3, padding=1, bias=False)
        )
    
    def forward(self, encoder_features):
        s0, e2, e3, e4, e5 = encoder_features
        c = self.bridge(e5)
        d1 = self.decoder1(c, e5)
        d2 = self.decoder2(d1, e4)
        d3 = self.decoder3(d2, e3)
        d4 = self.decoder4(d3, e2)
        d5 = self.decoder5(d4, s0)
        out = self.lastlayer(d5)
        return out


# 整合: ResNet34UNet #
class ResNet34UNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.encoder = ResNet34()
        self.decoder = UNetDecoder(num_classes=num_classes)
    
    def forward(self, x):
        encoder_features = self.encoder(x)
        out = self.decoder(encoder_features)
        return out

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ResNet34UNet(num_classes=2).to(device)

    # torchinfo 通常需要包含 Batch 維度，因此改成 (1, 3, 256, 256)
    summary(model, input_size=(1, 3, 256, 256), device=device)
