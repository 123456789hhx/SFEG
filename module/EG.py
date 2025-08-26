import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log
from module.CNR import CGR, Bott_CGR, CBR

class DEAM(nn.Module):
    def __init__(self, lchannel, hchannel):
        super(DEAM, self).__init__()
        self.reduce1 = CGR(lchannel, lchannel, 1, padding=0)
        self.reduce4 = CGR(hchannel, hchannel // 2, 1, padding=0)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(lchannel + hchannel//2, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, 1),
            nn.Sigmoid()
        )
        self.balance_factor = nn.Parameter(torch.ones(1) * 0.5)
        self.block = nn.Sequential(
            Bott_CGR(hchannel // 2 + lchannel, hchannel // 2, hchannel // 16, kernel_size=3, stride=1, padding=1),
            Bott_CGR(hchannel // 2, hchannel // 2, hchannel // 16, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(hchannel // 2, 1, 1))

    def forward(self, x1, x4):
        size = x1.size()[2:]
        x1 = self.reduce1(x1)
        x4 = self.reduce4(x4)
        x4 = F.interpolate(x4, size, mode='bilinear', align_corners=False)

        concat_features = torch.cat([x1, x4], dim=1)
        weights = self.channel_attention(concat_features)
        w1, w4 = weights[:, 0:1, :, :], weights[:, 1:2, :, :]

        weight_sum = w1 + w4
        scale = 2.0 / (weight_sum + 1e-8)

        x1 = x1 * (w1 * scale).expand_as(x1) * self.balance_factor
        x4 = x4 * (w4 * scale).expand_as(x4) * (1 - self.balance_factor)

        out = torch.cat((x4, x1), dim=1)
        out = self.block(out)

        return out


class EGFM(nn.Module):
    def __init__(self, dim):
        super(EGFM, self).__init__()
        t = int(abs((log(dim, 2) + 1) / 2))
        k = t if t % 2 else t + 1
        self.conv = Bott_CGR(dim, dim, dim // 8, kernel_size=3, stride=1, padding=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, c, att):
        if c.size() != att.size():
            att = F.interpolate(att, c.size()[2:], mode='bilinear', align_corners=False)
        x = c * att + c
        x = self.conv(x)
        w = self.avg_pool(x)
        w = self.mlp(w.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        w = self.sigmoid(w)
        x = x * w

        return x
    

class CSIM(nn.Module):
    def __init__(self, hchannel, channel):
        super(CSIM, self).__init__()
        self.conv1_1 = CBR(hchannel + channel, channel, 1, padding=0)
        self.conv3_1 = CBR(channel // 4, channel // 4, 3, padding=1)
        self.depthwise5_1 = CBR(channel // 4, channel // 4, 3, dilation=2, padding=2)
        self.depthwise7_1 = CBR(channel // 4, channel // 4, 3, dilation=3, padding=3)
        self.depthwise9_1 = CBR(channel // 4, channel // 4, 3, dilation=4, padding=4)
        self.conv1_2 = CBR(channel, channel, 1, padding=0)
        self.conv3_2 = CBR(channel, channel, 3, padding=1)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 16, channel, 1),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, lf, hf):
        if lf.size()[2:] != hf.size()[2:]:
            hf = F.interpolate(hf, size=lf.size()[2:], mode='bilinear', align_corners=False)

        fusion_weight = nn.Parameter(torch.ones(2)).to(lf.device)
        fusion_weight = F.softmax(fusion_weight, dim=0)
        x = torch.cat((lf * fusion_weight[0], hf * fusion_weight[1]), dim=1)
        x = self.conv1_1(x)
        xc = torch.chunk(x, 4, dim=1)

        x0 = self.conv3_1(xc[0] + xc[1])
        x1 = self.depthwise5_1(xc[1] + x0 + xc[2])
        x2 = self.depthwise7_1(xc[2] + x1 + xc[3])
        x3 = self.depthwise9_1(xc[3] + x2)
        xx = self.conv1_2(torch.cat((x0, x1, x2, x3), dim=1))
        x = self.conv3_2(x + xx)

        ca = self.channel_attention(x)
        x = x * ca

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial = torch.cat([avg_out, max_out], dim=1)
        spatial = self.spatial_attention(spatial)
        x = x * spatial + x

        return x