import torch
import torch.nn as nn
from module.CNR import Bott_Conv

class Fourier_Core(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Fourier_Core, self).__init__()
        self.conv = nn.Conv2d(in_channels * 2, out_channels * 2, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.GroupNorm(32, out_channels * 2)
        self.relu =nn.ReLU(inplace=True)

    def forward(self, x):
        batch, c, h, w = x.size()

        ffted = torch.fft.rfft2(x, norm='ortho')
        x_fft_real = torch.unsqueeze(torch.real(ffted), dim=-1)
        x_fft_imag = torch.unsqueeze(torch.imag(ffted), dim=-1)
        ffted = torch.cat((x_fft_real, x_fft_imag), dim=-1)

        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])

        ffted = self.conv(ffted)
        ffted = self.relu(self.bn(ffted))

        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:])
        ffted = ffted.permute(0, 1, 3, 4, 2).contiguous()
        ffted = torch.view_as_complex(ffted)

        output = torch.fft.irfft2(ffted, s=(h, w), norm='ortho')

        return output
    

class Fourier_Unit(nn.Module):
    def __init__(self, dim):
        super(Fourier_Unit, self).__init__()
        self.dim = dim
        self.conv_1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.GELU()
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.GELU()
        )
        self.fc = Fourier_Core(dim * 2, dim * 2)
        self.bn = nn.GroupNorm(32, dim * 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_1, x_2 = torch.split(x, self.dim, dim=1)
        x_1 = self.conv_1(x_1)
        x_2 = self.conv_2(x_2)
        x0 = torch.cat([x_1, x_2], dim=1)

        x = self.fc(x0) + x0
        x = self.relu(self.bn(x))

        return x
    

class Spatial_Fourier_Parallel_Mixer(nn.Module):
    def __init__(self, dim):
        super(Spatial_Fourier_Parallel_Mixer, self).__init__()
        self.dim = dim
        self.conv_init = nn.Sequential(
            nn.Conv2d(dim, dim * 2, 1),
            nn.GELU()
        )
        self.depthwise_1 = nn.Sequential(
            Bott_Conv(dim, dim, dim // 8, kernel_size=3, stride=1, padding=1),
            nn.GELU()
        )
        self.depthwise_2 = nn.Sequential(
            Bott_Conv(dim, dim, dim // 8, kernel_size=5, stride=1, padding=2),
            nn.GELU()
        )
        self.fu = Fourier_Unit(dim)
        self.conv_final = nn.Sequential(
            nn.Conv2d(dim * 4, dim * 2, 1),
            nn.GELU()
        )
        self.channel_attention_conv = nn.Sequential(
            nn.Conv2d(2 * dim, dim, 1),
            Bott_Conv(dim, dim, dim // 8, kernel_size=3, stride=1, padding=1),
            nn.GELU()
        )
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim // 4, dim, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_init(x)
        x_global = self.fu(x)
        x_1, x_2 = torch.split(x, self.dim, dim=1)
        x_local_1 = self.depthwise_1(x_1)
        x_local_2 = self.depthwise_2(x_2)
        x_local = torch.cat([x_local_1, x_local_2], dim=1)
        x = torch.cat([x_global, x_local], dim=1)
        x = self.conv_final(x)
        x = self.channel_attention_conv(x)
        x = self.channel_attention(x) * x

        return x