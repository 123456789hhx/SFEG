import torch
import torch.nn as nn
from module.CNR import Bott_Conv

class Prior_Feed_Gate_Mixer(nn.Module):
    def __init__(self, dim, scale_ratio=2, spilt_num=4):
        super(Prior_Feed_Gate_Mixer, self).__init__()
        self.dim = dim
        self.dim_sp = dim * scale_ratio // spilt_num
        self.conv_init = nn.Sequential(
            nn.Conv2d(dim, dim * 2, 1),
            nn.GELU()
        )
        self.dw_conv = nn.Sequential(
            Bott_Conv(dim * 2, dim * 2, dim // 4, kernel_size=3, stride=1, padding=1),
            nn.GELU()
        )
        self.mask_in = nn.Sequential(
            nn.Conv2d(1, self.dim_sp, 1),
            nn.GELU()
        )
        self.mask_dw_conv_1 = nn.Sequential(
            Bott_Conv(self.dim_sp // 2, 1, self.dim_sp // 4, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        self.mask_dw_conv_2 = nn.Sequential(
            Bott_Conv(self.dim_sp // 2, 1, self.dim_sp // 4, kernel_size=5, stride=1, padding=2),
            nn.Sigmoid()
        )
        self.mask_out = nn.Sequential(
            nn.Conv2d(2, 1, 1),
            nn.GELU()
        )
        self.conv_final = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
            nn.GELU()
        )

    def forward(self, x, mask):
        x = self.conv_init(x)
        x = self.dw_conv(x)
        x = list(torch.split(x, self.dim, dim=1))
        mask = self.mask_in(mask)
        mask = list(torch.split(mask, self.dim_sp // 2, dim=1))
        mask[0] = self.mask_dw_conv_1(mask[0])
        mask[1] = self.mask_dw_conv_2(mask[1])
        x[0] = mask[0] * x[0]
        x[1] = mask[1] * x[1]
        x = torch.cat(x, dim=1)
        x = self.conv_final(x)
        mask = self.mask_out(torch.cat(mask, dim=1))

        return x, mask