import torch
import torch.nn as nn
import torch.nn.functional as F
class Conv_Layers(nn.Module):
    '''conv -> BN -> ReLU * 2 '''
    def __init__(self, in_channels, out_channels, num_groups=8):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups= num_groups, num_channels=out_channels), 
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm3d(out_channels),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_layers(x)
    

class Down_Sample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            Conv_Layers(in_channels, out_channels)
        )

    def forward(self, x):
        return self.encoder(x)

class Up_Sample(nn.Module):
    def __init__(self, in_channels, out_channels, trilinear = True):
        super().__init__()
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        
        self.conv = Conv_Layers(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2, diffZ // 2, diffZ - diffZ // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    

class Out(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size = 1)

    def forward(self, x):
        return self.conv(x)


class BraTS_Model(nn.Module):
    '''UNet Model Archietecture'''
    def __init__(self, in_channels, n_classes, n_channels):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.n_channels = n_channels

        self.conv = Conv_Layers(in_channels, n_channels)
        self.enc1 = Down_Sample(n_channels, 2 * n_channels)
        self.enc2 = Down_Sample(2 * n_channels, 4 * n_channels)
        self.enc3 = Down_Sample(4 * n_channels, 8 * n_channels)
        self.enc4 = Down_Sample(8 * n_channels, 8 * n_channels)

        self.dec1 = Up_Sample(16 * n_channels, 4 * n_channels)
        self.dec2 = Up_Sample(8 * n_channels, 2 * n_channels)
        self.dec3 = Up_Sample(4 * n_channels, n_channels)
        self.dec4 = Up_Sample(2 * n_channels, n_channels)
        self.out = Out(n_channels, n_classes)

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.enc1(x1)
        x3 = self.enc2(x2)
        x4 = self.enc3(x3)
        x5 = self.enc4(x4)

        mask = self.dec1(x5, x4)
        mask = self.dec2(mask, x3)
        mask = self.dec3(mask, x2)
        mask = self.dec4(mask, x1)
        mask = self.out(mask)
        return mask