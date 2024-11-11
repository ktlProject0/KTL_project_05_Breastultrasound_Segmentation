import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34

class ResNetUNetEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNetUNetEncoder, self).__init__()
        self.resnet = resnet34(pretrained=pretrained)
        self.enc1 = nn.Sequential(self.resnet.conv1, self.resnet.bn1, self.resnet.relu)
        self.enc2 = nn.Sequential(self.resnet.maxpool, self.resnet.layer1)
        self.enc3 = self.resnet.layer2
        self.enc4 = self.resnet.layer3
        self.enc5 = self.resnet.layer4

    def forward(self, x):
        skip_connections = []
        x = self.enc1(x)
        skip_connections.append(x)
        x = self.enc2(x)
        skip_connections.append(x)
        x = self.enc3(x)
        skip_connections.append(x)
        x = self.enc4(x)
        skip_connections.append(x)
        x = self.enc5(x)
        return x, skip_connections

class ResNetUNetDecoder(nn.Module):
    def __init__(self, n_classes):
        super(ResNetUNetDecoder, self).__init__()
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder4 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder3 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.upconv1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.decoder1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.final_conv = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x, skip_connections):
        x = self.upconv4(x)
        x = torch.cat([x, skip_connections[3]], dim=1)
        x = self.decoder4(x)
        x = self.upconv3(x)
        x = torch.cat([x, skip_connections[2]], dim=1)
        x = self.decoder3(x)
        x = self.upconv2(x)
        x = torch.cat([x, skip_connections[1]], dim=1)
        x = self.decoder2(x)
        x = self.upconv1(x)
        x = torch.cat([x, skip_connections[0]], dim=1)
        x = self.decoder1(x)
        x = self.final_conv(x)
        return x

class Net(nn.Module):
    def __init__(self, n_classes):
        super(Net, self).__init__()
        self.encoder = ResNetUNetEncoder()
        self.decoder = ResNetUNetDecoder(n_classes)
        self.activation = nn.Sigmoid() if n_classes == 1 else nn.Softmax(dim=1)

    def forward(self, x):
        x, skip_connections = self.encoder(x)
        x = self.decoder(x, skip_connections)
        x = F.interpolate(x, size=(128, 128), mode='bilinear', align_corners=False)
        return self.activation(x)
