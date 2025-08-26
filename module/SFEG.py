import torch
import torch.nn as nn
from module.EG import *
from module.SFA import SFA_Backbone
from module.CNR import CGR
# from thop import profile

class SFEG(nn.Module):
    def __init__(self, embed_dim=[32, 64, 128, 256, 512], depth=[3, 3, 3, 3, 3]):
        super(SFEG, self).__init__()
        self.encoder = SFA_Backbone(
            in_chans=3,
            patch_size=1,
            embed_dim=embed_dim,
            depth=depth,
            embed_kernel_size=3
        )

        self.cgr1 = CGR(embed_dim[0], embed_dim[1], kernel_size=3, stride=1, padding=1)
        self.cgr2 = CGR(embed_dim[1], embed_dim[2], kernel_size=3, stride=1, padding=1)
        self.cgr3 = CGR(embed_dim[2], embed_dim[3], kernel_size=3, stride=1, padding=1)
        self.cgr4 = CGR(embed_dim[3], embed_dim[4], kernel_size=3, stride=1, padding=1)

        self.eam = DEAM(embed_dim[1], embed_dim[4])

        self.efm1 = EGFM(embed_dim[1])
        self.efm2 = EGFM(embed_dim[2])
        self.efm3 = EGFM(embed_dim[3])
        self.efm4 = EGFM(embed_dim[4])

        self.conv_1 = CGR(embed_dim[1], embed_dim[1], kernel_size=1, stride=1, padding=0)
        self.conv_2 = CGR(embed_dim[2], embed_dim[2], kernel_size=1, stride=1, padding=0)
        self.conv_3 = CGR(embed_dim[3], embed_dim[3], kernel_size=1, stride=1, padding=0)
        self.conv_4 = CGR(embed_dim[4], embed_dim[4], kernel_size=1, stride=1, padding=0)

        self.cam1 = CSIM(embed_dim[2], embed_dim[1])
        self.cam2 = CSIM(embed_dim[3], embed_dim[2])
        self.cam3 = CSIM(embed_dim[4], embed_dim[3])

        self.predictor1 = nn.Conv2d(embed_dim[1], 1, 1)
        self.predictor2 = nn.Conv2d(embed_dim[2], 1, 1)
        self.predictor3 = nn.Conv2d(embed_dim[3], 1, 1)

    def forward(self, x):
        x_list = self.encoder(x)
        x1 = x_list[0]
        x2 = x_list[1]
        x3 = x_list[2]
        x4 = x_list[3]

        x1 = self.cgr1(x1)
        x2 = self.cgr2(x2)
        x3 = self.cgr3(x3)
        x4 = self.cgr4(x4)
        
        edge = self.eam(x1, x4)
        edge_att = torch.sigmoid(edge)

        x1_att = self.efm1(x1, edge_att)
        x2_att = self.efm2(x2, edge_att)
        x3_att = self.efm3(x3, edge_att)
        x4_att = self.efm4(x4, edge_att)

        x1r = self.conv_1(x1_att)
        x2r = self.conv_2(x2_att)
        x3r = self.conv_3(x3_att)
        x4r = self.conv_4(x4_att)

        x34 = self.cam3(x3r, x4r)
        x234 = self.cam2(x2r, x34)
        x1234 = self.cam1(x1r, x234)

        o3 = self.predictor3(x34)
        o3 = F.interpolate(o3, size=x.shape[2:], mode='bilinear', align_corners=False)
        o2 = self.predictor2(x234)
        o2 = F.interpolate(o2, size=x.shape[2:], mode='bilinear', align_corners=False)
        o1 = self.predictor1(x1234)
        o1 = F.interpolate(o1, size=x.shape[2:], mode='bilinear', align_corners=False)

        return o1, o2, o3


# def test():
#     x = torch.randn((1, 3, 256, 256)).cuda()

#     model = SFEG().cuda()
#     flops, params = profile(model, (x,))
#     print('flops: %.2f G, params: %.2f M' % (flops / 1e9, params / 1e6))
#     # preds = model(x)
#     # print(x.shape)
#     # print(preds[0].shape, preds[1].shape, preds[2].shape)


# if __name__ == "__main__":
#     test()
