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
        features = input.features
        newTensor = ME.SparseTensor(features = features*self.scalar,
                                    coordinate_map_key = input.coordinate_map_key,
                                    coordinate_manager = input.coordinate_manager,
                                    )
        return newTensor

class FeatureSelect(nn.Module):
    def __init__(self, featureColumns):
        super(FeatureSelect, self).__init__()

        self.featureColumns = [int(i) for i in featureColumns]

    def forward(self, input):
        # if len(self.featureColumns) == 1:
        #     selectedFeatures = input.features[:,self.featureColumns[0]]
        # else:
        selectedFeatures = input.features[:,self.featureColumns]
        newTensor = ME.SparseTensor(features = selectedFeatures,
                                    coordinate_map_key = input.coordinate_map_key,
                                    coordinate_manager = input.coordinate_manager,
                                    )
        return newTensor

class Threshold(nn.Module):
    def __init__(self, thresholdValue, featureColumn):
        super(Threshold, self).__init__()

        self.thresholdValue = thresholdValue
        self.featureColumn = int(featureColumn)

    def forward(self, input):
        cutFeature = input.features[:, self.featureColumn]
        thresholdMask = ( cutFeature > self.thresholdValue )

        newTensor = ME.SparseTensor(features = input.features[thresholdMask],
                                    coordinates = input.coordinates[thresholdMask],
                                    )
        return newTensor

class ResNetBlock(torch.nn.Module):
    def __init__(self, in_features, out_features, kernel_size, name = 'resBlock'):
        super(ResNetBlock, self).__init__()

        if in_features != out_features:
            self.residual = ME.MinkowskiLinear(in_features, out_features)
        else:
            self.residual = Identity()
        
        self.norm1 = ME.MinkowskiBatchNorm(in_features)
        self.act1 = ME.MinkowskiReLU()
        self.conv1 = ME.MinkowskiConvolution(in_channels = in_features,
                                             out_channels = out_features,
                                             kernel_size = kernel_size,
                                             stride = 1,
                                             dimension = 3)

        self.norm2 = ME.MinkowskiBatchNorm(out_features)
        self.act2 = ME.MinkowskiReLU()
        self.conv2 = ME.MinkowskiConvolution(in_channels = out_features,
                                             out_channels = out_features,
                                             kernel_size = kernel_size,
                                             stride = 1,
                                             dimension = 3)
        
    def forward(self, x):

        residual = self.residual(x)
        
        out = self.conv1(self.act1(self.norm1(x)))
        out = self.conv2(self.act2(self.norm2(out)))
        out += residual

        return out

class DropoutBlock(torch.nn.Module):
    def __init__(self, in_features, out_features, kernel_size, name = 'DropoutBlock'):
        super(DropoutBlock, self).__init__()

        self.conv1 = ME.MinkowskiConvolution(in_channels = in_features,
                                             out_channels = out_features,
                                             kernel_size = kernel_size,
                                             stride = 1,
                                             dimension = 3)
        self.dropout1 = ME.MinkowskiDropout()
        self.norm1 = ME.MinkowskiBatchNorm(out_features)
        self.act1 = ME.MinkowskiReLU()

        self.conv2 = ME.MinkowskiConvolution(in_channels = out_features,
                                             out_channels = out_features,
                                             kernel_size = kernel_size,
                                             stride = 1,
                                             dimension = 3)
        self.dropout2 = ME.MinkowskiDropout()
        self.norm2 = ME.MinkowskiBatchNorm(out_features)
        self.act2 = ME.MinkowskiReLU()
        
    def forward(self, x):

        out = self.act1(self.norm1(self.dropout1(self.conv1(x))))
        out = self.act2(self.norm2(self.dropout2(self.conv2(out))))

        return out

class DownSample(torch.nn.Module):
    def __init__(self, in_features, out_features, kernel_size, dropout = False, name = 'DownSample'):
        super(DownSample, self).__init__()
        
        self.norm = ME.MinkowskiBatchNorm(in_features)
        self.act = ME.MinkowskiReLU()
        self.conv = ME.MinkowskiConvolution(in_channels = in_features,
                                            out_channels = out_features,
                                            kernel_size = kernel_size,
                                            stride = kernel_size,
                                            dimension = 3)

        self.useDropout = dropout
        self.dropout = ME.MinkowskiDropout()
        
    def forward(self, x):
        
        out = self.conv(self.act(self.norm(x)))
        if self.useDropout:
            out = self.dropout(out)

        return out

class UpSample(torch.nn.Module):
    def __init__(self, in_features, out_features, kernel_size, dropout = False, name = 'DownSample'):
        super(UpSample, self).__init__()
        
        self.norm = ME.MinkowskiBatchNorm(in_features)
        self.act = ME.MinkowskiReLU()
        self.conv = ME.MinkowskiConvolutionTranspose(
            in_channels = in_features,
            out_channels = out_features,
            kernel_size = kernel_size,
            stride = kernel_size,
            dimension = 3,
        )

        self.useDropout = dropout
        self.dropout = ME.MinkowskiDropout()
        
    def forward(self, x, cmk = None):
        
        if cmk:
            out = self.conv(self.act(self.norm(x)), cmk)
        else:
            out = self.conv(self.act(self.norm(x)))
        if self.useDropout:
            out = self.dropout(out)

        return out
