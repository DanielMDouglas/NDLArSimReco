import torch
import torch.nn as nn
import MinkowskiEngine as ME

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, input):
        return input

class Scaling(nn.Module):
    def __init__(self, scalingFactor):
        super(Scaling, self).__init__()

        self.scalar = scalingFactor

    def forward(self, input):
        input.features[:] *= self.scalar
        return input

class ResNetBlock(torch.nn.Module):
    def __init__(self, in_features, out_features, name = 'resBlock'):
        super(ResNetBlock, self).__init__()

        if in_features != out_features:
            self.residual = ME.MinkowskiLinear(in_features, out_features)
        else:
            self.residual = Identity()
        
        self.conv1 = ME.MinkowskiConvolution(in_channels = in_features,
                                             out_channels = out_features,
                                             kernel_size = 3,
                                             stride = 1,
                                             dimension = 3)
        self.act1 = ME.MinkowskiReLU()
        self.norm1 = ME.MinkowskiBatchNorm(in_features)
        # self.norm1 = Identity()
        self.conv2 = ME.MinkowskiConvolution(in_channels = out_features,
                                             out_channels = out_features,
                                             kernel_size = 3,
                                             stride = 1,
                                             dimension = 3)
        self.act2 = ME.MinkowskiReLU()
        self.norm2 = ME.MinkowskiBatchNorm(out_features)
        # self.norm2 = Identity()
        
    def forward(self, x):

        residual = self.residual(x)
        
        out = self.conv1(self.act1(self.norm1(x)))
        out = self.conv2(self.act2(self.norm2(out)))
        out += residual

        return out
