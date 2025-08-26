import torch
import torch.nn as nn
from module.SFPM import Spatial_Fourier_Parallel_Mixer
from module.PFGM import Prior_Feed_Gate_Mixer
# from thop import profile

class PatchEmbed(nn.Module):
    def __init__(self, in_chans=3, patch_size=1, embed_dim=32, kernel_size=3):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size,
                              padding=(kernel_size - patch_size + 1) // 2)

    def forward(self, x):
        x = self.proj(x)
        return x


class DownSample(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.proj = nn.Sequential(nn.Conv2d(input_dim, input_dim // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                    nn.PixelUnshuffle(2))
        
    def forward(self, x):
        x = self.proj(x)
        return x


class DownSample_CP(nn.Module):
    def __init__(self, input_dim=4, output_dim=1, kernel_size=3, stride=1):
        super().__init__()
        self.proj = nn.Sequential(nn.PixelUnshuffle(2),
                                  nn.Conv2d(input_dim, output_dim, kernel_size, stride, padding=1, bias=False))

    def forward(self, x):
        x = self.proj(x)
        return x


class SFA_Block(nn.Module):
    def __init__(self, dim):
        super(SFA_Block, self).__init__()
        self.dim = dim
        self.norm1 = nn.GroupNorm(32, dim)
        self.norm2 = nn.GroupNorm(32, dim)
        self.mixer = Spatial_Fourier_Parallel_Mixer(dim)
        self.ffn = Prior_Feed_Gate_Mixer(dim)

    def forward(self, mix_input):
        x, mask = mix_input
        copy = x
        x = self.norm1(x)
        x = self.mixer(x)
        x = x + copy

        copy = x
        x = self.norm2(x)
        x, mask = self.ffn(x, mask)
        x = x + copy

        return x, mask


class SFA_Stage(nn.Module):
    def __init__(self, depth=int, in_channels=int):
        super(SFA_Stage, self).__init__()
        self.blocks = nn.Sequential(*[SFA_Block(dim=in_channels) for _ in range(depth)])

    def forward(self, mix_input):
        output = self.blocks(mix_input)
        return output


def get_gradient_prior(tensor):
    if tensor.dim() == 4 and tensor.size(1) > 1:
        tensor = tensor.mean(dim=1, keepdim=True)
        
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                           dtype=torch.float32).reshape(1, 1, 3, 3).to(tensor.device)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                           dtype=torch.float32).reshape(1, 1, 3, 3).to(tensor.device)
    
    grad_x = nn.functional.conv2d(tensor, sobel_x, padding=1)
    grad_y = nn.functional.conv2d(tensor, sobel_y, padding=1)
    
    grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
    
    return grad_magnitude


class SFA_Backbone(nn.Module):
    def __init__(self, in_chans=3, patch_size=1, embed_dim=[32, 64, 128, 256, 512], depth=[3, 3, 3, 3, 3], 
                 embed_kernel_size=3):
        super(SFA_Backbone, self).__init__()
        self.patch_size = patch_size

        self.patch_embed = PatchEmbed(in_chans=in_chans, patch_size=patch_size,
                                      embed_dim=embed_dim[0], kernel_size=embed_kernel_size)
        
        self.layer1 = SFA_Stage(depth=depth[0], in_channels=embed_dim[0])
        self.downsample1 = DownSample(input_dim=embed_dim[0])
        self.down_rcp1 = DownSample_CP()

        self.layer2 = SFA_Stage(depth=depth[1], in_channels=embed_dim[1])
        self.downsample2 = DownSample(input_dim=embed_dim[1])
        self.down_rcp2 = DownSample_CP()

        self.layer3 = SFA_Stage(depth=depth[2], in_channels=embed_dim[2])
        self.downsample3 = DownSample(input_dim=embed_dim[2])
        self.down_rcp3 = DownSample_CP()

        self.layer4 = SFA_Stage(depth=depth[3], in_channels=embed_dim[3])
        self.downsample4 = DownSample(input_dim=embed_dim[3])
        self.down_rcp4 = DownSample_CP()

    def forward(self, x):
        layer_output = []
        layer_after_downsample_output = []

        mask = get_gradient_prior(x)
        x = self.patch_embed(x)

        x, mask = self.layer1((x, mask))
        layer_output.append(x)
        x = self.downsample1(x)
        layer_after_downsample_output.append(x)
        mask = self.down_rcp1(mask)

        x, mask = self.layer2((x, mask))
        layer_output.append(x)
        x = self.downsample2(x)
        layer_after_downsample_output.append(x)
        mask = self.down_rcp2(mask)

        x, mask = self.layer3((x, mask))
        layer_output.append(x)
        x = self.downsample3(x)
        layer_after_downsample_output.append(x)
        mask = self.down_rcp3(mask)

        x, mask = self.layer4((x, mask))
        layer_output.append(x)
        x = self.downsample4(x)
        layer_after_downsample_output.append(x)
        mask = self.down_rcp4(mask)

        return layer_output
    

# def test():
#     x = torch.randn((1, 3, 256, 256)).cuda()

#     model = SFA_Backbone().cuda()
#     flops, params = profile(model, (x,))
#     print('flops: %.2f G, params: %.2f M' % (flops / 1e9, params / 1e6))
#     preds = model(x)
#     print(x.shape)
#     print(preds[0].shape, preds[1].shape, preds[2].shape)


# if __name__ == "__main__":
#     test()