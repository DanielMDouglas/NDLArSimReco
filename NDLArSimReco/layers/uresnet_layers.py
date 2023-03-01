import torch
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import MinkowskiEngine as ME
ME.set_sparse_tensor_operation_mode(ME.SparseTensorOperationMode.SHARE_COORDINATE_MANAGER)

from NDLArSimReco.layers.blocks import ResNetBlock, Identity

class UNet(torch.nn.Module):
    def __init__(self, in_features, out_features, depth = 2, nFilters = 16, name='unet'):
        super(UNet, self).__init__()

        self.depth = depth # number of pool/unpool layers, not including input + output
        self.nFilters = nFilters
        self.in_features = in_features
        self.out_features = out_features
        
        self.input_block = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels = self.in_features,
                out_channels = self.nFilters,
                kernel_size = 3,
                stride = 1,
                dimension = 3,
            ), 
        )

        self.featureSizesEnc = [(self.nFilters*2**i, self.nFilters*2**(i+1))
                                for i in range(self.depth)]
        self.featureSizesDec = [(out_feat, in_feat)  
                                for in_feat, out_feat in self.featureSizesEnc]
        self.featureSizesDec.reverse()
        
        self.encoding_layers = []
        self.decoding_layers = []

        self.encoding_blocks = []
        self.decoding_blocks = []
        
        for i in range(self.depth):
            self.encoding_layers.append(
                ME.MinkowskiConvolution(
                    in_channels = self.featureSizesEnc[i][0],
                    out_channels = self.featureSizesEnc[i][1],
                    kernel_size = 2,
                    stride = 2,
                    dimension = 3)
            )
            self.encoding_blocks.append(
                nn.Sequential(
                    ME.MinkowskiConvolution(
                        in_channels = self.featureSizesEnc[i][1],
                        out_channels = self.featureSizesEnc[i][1],
                        kernel_size = 3,
                        stride = 1,
                        dimension = 3),
                    ME.MinkowskiBatchNorm(self.featureSizesEnc[i][1]),
                    ME.MinkowskiReLU(),
                    ME.MinkowskiConvolution(
                        in_channels = self.featureSizesEnc[i][1],
                        out_channels = self.featureSizesEnc[i][1],
                        kernel_size = 3,
                        stride = 1,
                        dimension = 3),
                    ME.MinkowskiBatchNorm(self.featureSizesEnc[i][1]),
                    ME.MinkowskiReLU(),
                )
            )
        self.encoding_layers = nn.Sequential(*self.encoding_layers)
        self.encoding_blocks = nn.Sequential(*self.encoding_blocks)

        for i in range(self.depth):
            self.decoding_layers.append(
                ME.MinkowskiConvolutionTranspose(
                    in_channels = self.featureSizesDec[i][0],
                    out_channels = self.featureSizesDec[i][1],
                    kernel_size = 2,
                    stride = 2,
                    dimension = 3)
            )
            self.decoding_blocks.append(
                nn.Sequential(
                    ME.MinkowskiConvolution(
                        in_channels = 2*self.featureSizesDec[i][1],
                        out_channels = 2*self.featureSizesDec[i][1],
                        kernel_size = 3,
                        stride = 1,
                        dimension = 3),
                    ME.MinkowskiBatchNorm(2*self.featureSizesDec[i][1]),
                    ME.MinkowskiReLU(),
                    ME.MinkowskiConvolution(
                        in_channels = 2*self.featureSizesDec[i][1],
                        out_channels = self.featureSizesDec[i][1],
                        kernel_size = 3,
                        stride = 1,
                        dimension = 3),
                    ME.MinkowskiBatchNorm(self.featureSizesDec[i][1]),
                    ME.MinkowskiReLU(),
                )
            )
        self.decoding_layers = nn.Sequential(*self.decoding_layers)
        self.decoding_blocks = nn.Sequential(*self.decoding_blocks)

        self.output_block = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels = self.featureSizesDec[-1][1],
                out_channels = self.featureSizesDec[-1][1],
                kernel_size = 3,
                stride = 1,
                dimension = 3,
            ),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(
                in_channels = self.featureSizesDec[-1][1],
                out_channels = self.out_features,
                kernel_size = 3,
                stride = 1,
                dimension = 3,
            ),
        )
        
    def forward(self, x):
        encodingFeatures = []
        coordKeys = []

        out = self.input_block(x)
        for i in range(self.depth):
            encodingFeatures.append(out)
            coordKeys.append(out.coordinate_map_key)

            out = self.encoding_layers[i](out)
            out = self.encoding_blocks[i](out)

        for i in range(self.depth):
            skip = encodingFeatures[-i -1]
            cmk = coordKeys[-i -1]
            
            out = self.decoding_layers[i](out, cmk)
            out = ME.cat(out, skip)
            out = self.decoding_blocks[i](out)

        out = self.output_block(out)
        
        return out

class UNet_dropout(torch.nn.Module):
    def __init__(self, in_features, out_features, depth = 2, nFilters = 16, name='unet_dropout'):
        super(UNet_dropout, self).__init__()

        self.depth = depth # number of pool/unpool layers, not including input + output
        self.nFilters = nFilters
        self.in_features = in_features
        self.out_features = out_features
        
        self.input_block = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels = self.in_features,
                out_channels = self.nFilters,
                kernel_size = 3,
                stride = 1,
                dimension = 3,
            ), 
        )

        self.featureSizesEnc = [(self.nFilters*2**i, self.nFilters*2**(i+1))
                                for i in range(self.depth)]
        self.featureSizesDec = [(out_feat, in_feat)  
                                for in_feat, out_feat in self.featureSizesEnc]
        self.featureSizesDec.reverse()
        
        self.encoding_layers = []
        self.decoding_layers = []

        self.encoding_blocks = []
        self.decoding_blocks = []
        
        for i in range(self.depth):
            self.encoding_layers.append(
                ME.MinkowskiConvolution(
                    in_channels = self.featureSizesEnc[i][0],
                    out_channels = self.featureSizesEnc[i][1],
                    kernel_size = 2,
                    stride = 2,
                    dimension = 3)
            )
            self.encoding_blocks.append(
                nn.Sequential(
                    ME.MinkowskiDropout(),
                    ME.MinkowskiConvolution(
                        in_channels = self.featureSizesEnc[i][1],
                        out_channels = self.featureSizesEnc[i][1],
                        kernel_size = 3,
                        stride = 1,
                        dimension = 3),
                    ME.MinkowskiBatchNorm(self.featureSizesEnc[i][1]),
                    ME.MinkowskiReLU(),
                    ME.MinkowskiConvolution(
                        in_channels = self.featureSizesEnc[i][1],
                        out_channels = self.featureSizesEnc[i][1],
                        kernel_size = 3,
                        stride = 1,
                        dimension = 3),
                    ME.MinkowskiBatchNorm(self.featureSizesEnc[i][1]),
                    ME.MinkowskiReLU(),
                )
            )
        self.encoding_layers = nn.Sequential(*self.encoding_layers)
        self.encoding_blocks = nn.Sequential(*self.encoding_blocks)

        for i in range(self.depth):
            self.decoding_layers.append(
                ME.MinkowskiConvolutionTranspose(
                    in_channels = self.featureSizesDec[i][0],
                    out_channels = self.featureSizesDec[i][1],
                    kernel_size = 2,
                    stride = 2,
                    dimension = 3)
            )
            self.decoding_blocks.append(
                nn.Sequential(
                    ME.MinkowskiConvolution(
                        in_channels = 2*self.featureSizesDec[i][1],
                        out_channels = 2*self.featureSizesDec[i][1],
                        kernel_size = 3,
                        stride = 1,
                        dimension = 3),
                    ME.MinkowskiBatchNorm(2*self.featureSizesDec[i][1]),
                    ME.MinkowskiReLU(),
                    ME.MinkowskiConvolution(
                        in_channels = 2*self.featureSizesDec[i][1],
                        out_channels = self.featureSizesDec[i][1],
                        kernel_size = 3,
                        stride = 1,
                        dimension = 3),
                    ME.MinkowskiBatchNorm(self.featureSizesDec[i][1]),
                    ME.MinkowskiReLU(),
                    ME.MinkowskiDropout(),
                )
            )
        self.decoding_layers = nn.Sequential(*self.decoding_layers)
        self.decoding_blocks = nn.Sequential(*self.decoding_blocks)

        self.output_block = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels = self.featureSizesDec[-1][1],
                out_channels = self.featureSizesDec[-1][1],
                kernel_size = 3,
                stride = 1,
                dimension = 3,
            ),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(
                in_channels = self.featureSizesDec[-1][1],
                out_channels = self.out_features,
                kernel_size = 3,
                stride = 1,
                dimension = 3,
            ),
        )
        
    def forward(self, x):
        encodingFeatures = []
        coordKeys = []

        out = self.input_block(x)
        for i in range(self.depth):
            encodingFeatures.append(out)
            coordKeys.append(out.coordinate_map_key)

            out = self.encoding_layers[i](out)
            out = self.encoding_blocks[i](out)

        for i in range(self.depth):
            skip = encodingFeatures[-i -1]
            cmk = coordKeys[-i -1]
            
            out = self.decoding_layers[i](out, cmk)
            out = ME.cat(out, skip)
            out = self.decoding_blocks[i](out)

        out = self.output_block(out)
        
        return out

class UResNet(torch.nn.Module):
    def __init__(self, in_features, out_features, depth = 2, nFilters = 16, name='uresnet'):
        super(UResNet, self).__init__()

        self.depth = depth # number of pool/unpool layers, not including input + output
        self.nFilters = nFilters
        self.in_features = in_features
        self.out_features = out_features
        
        self.input_block = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels = self.in_features,
                out_channels = self.nFilters,
                kernel_size = 3,
                stride = 1,
                dimension = 3,
            ) 
        )

        self.featureSizesEnc = [(self.nFilters*2**i, self.nFilters*2**(i+1))
                                for i in range(self.depth)]
        self.featureSizesDec = [(out_feat, in_feat)  
                                for in_feat, out_feat in self.featureSizesEnc]
        self.featureSizesDec.reverse()
        
        self.encoding_layers = []
        self.decoding_layers = []

        self.encoding_blocks = []
        self.decoding_blocks = []
        
        for i in range(self.depth):
            self.encoding_layers.append(
                ME.MinkowskiConvolution(
                    in_channels = self.featureSizesEnc[i][0],
                    out_channels = self.featureSizesEnc[i][1],
                    kernel_size = 2,
                    stride = 2,
                    dimension = 3)
            )
            self.encoding_blocks.append(
                ResNetBlock(self.featureSizesEnc[i][1],
                            self.featureSizesEnc[i][1])
            )
        self.encoding_layers = nn.Sequential(*self.encoding_layers)
        self.encoding_blocks = nn.Sequential(*self.encoding_blocks)

        for i in range(self.depth):
            self.decoding_layers.append(
                ME.MinkowskiConvolutionTranspose(
                    in_channels = self.featureSizesDec[i][0],
                    out_channels = self.featureSizesDec[i][1],
                    kernel_size = 2,
                    stride = 2,
                    dimension = 3)
            )
            self.decoding_blocks.append(
                ResNetBlock(2*self.featureSizesDec[i][1],
                            self.featureSizesDec[i][1])
            )
        self.decoding_layers = nn.Sequential(*self.decoding_layers)
        self.decoding_blocks = nn.Sequential(*self.decoding_blocks)

        self.output_block = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels = self.featureSizesDec[-1][1],
                out_channels = self.featureSizesDec[-1][1],
                kernel_size = 3,
                stride = 1,
                dimension = 3,
            ),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(
                in_channels = self.featureSizesDec[-1][1],
                out_channels = self.out_features,
                kernel_size = 3,
                stride = 1,
                dimension = 3,
            ),
        )
        
    def forward(self, x):
        encodingFeatures = []
        coordKeys = []

        out = self.input_block(x)
        for i in range(self.depth):
            encodingFeatures.append(Identity()(out))
            coordKeys.append(out.coordinate_map_key)

            out = self.encoding_layers[i](out)
            out = self.encoding_blocks[i](out)

        for i in range(self.depth):
            skip = encodingFeatures[-i -1]
            cmk = coordKeys[-i -1]
            
            out = self.decoding_layers[i](out, cmk)
            out = ME.cat(out, skip)
            out = self.decoding_blocks[i](out)

        out = self.output_block(out)
        
        return out

class UResNet_dropout(torch.nn.Module):
    def __init__(self, in_features, out_features, depth = 2, nFilters = 16, name='uresnet_dropout'):
        super(UResNet_dropout, self).__init__()

        self.depth = depth # number of pool/unpool layers, not including input + output
        self.nFilters = nFilters
        self.in_features = in_features
        self.out_features = out_features
        
        self.input_block = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels = self.in_features,
                out_channels = self.nFilters,
                kernel_size = 3,
                stride = 1,
                dimension = 3,
            ) 
        )

        self.featureSizesEnc = [(self.nFilters*2**i, self.nFilters*2**(i+1))
                                for i in range(self.depth)]
        self.featureSizesDec = [(out_feat, in_feat)  
                                for in_feat, out_feat in self.featureSizesEnc]
        self.featureSizesDec.reverse()
        
        self.encoding_layers = []
        self.decoding_layers = []

        self.encoding_blocks = []
        self.decoding_blocks = []
        
        for i in range(self.depth):
            self.encoding_layers.append(
                ME.MinkowskiConvolution(
                    in_channels = self.featureSizesEnc[i][0],
                    out_channels = self.featureSizesEnc[i][1],
                    kernel_size = 2,
                    stride = 2,
                    dimension = 3)
            )
            self.encoding_blocks.append(
                nn.Sequential(
                    ME.MinkowskiDropout(),
                    ResNetBlock(self.featureSizesEnc[i][1],
                                self.featureSizesEnc[i][1]),
                )
            )
        self.encoding_layers = nn.Sequential(*self.encoding_layers)
        self.encoding_blocks = nn.Sequential(*self.encoding_blocks)

        for i in range(self.depth):
            self.decoding_layers.append(
                ME.MinkowskiConvolutionTranspose(
                    in_channels = self.featureSizesDec[i][0],
                    out_channels = self.featureSizesDec[i][1],
                    kernel_size = 2,
                    stride = 2,
                    dimension = 3)
            )
            self.decoding_blocks.append(
                nn.Sequential(
                    ResNetBlock(2*self.featureSizesDec[i][1],
                                self.featureSizesDec[i][1]),
                    ME.MinkowskiDropout(),
                )
            )
        self.decoding_layers = nn.Sequential(*self.decoding_layers)
        self.decoding_blocks = nn.Sequential(*self.decoding_blocks)

        self.output_block = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels = self.featureSizesDec[-1][1],
                out_channels = self.featureSizesDec[-1][1],
                kernel_size = 3,
                stride = 1,
                dimension = 3,
            ),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(
                in_channels = self.featureSizesDec[-1][1],
                out_channels = self.out_features,
                kernel_size = 3,
                stride = 1,
                dimension = 3,
            ),
        )
        
    def forward(self, x):
        encodingFeatures = []
        coordKeys = []

        out = self.input_block(x)
        for i in range(self.depth):
            encodingFeatures.append(Identity()(out))
            coordKeys.append(out.coordinate_map_key)

            out = self.encoding_layers[i](out)
            out = self.encoding_blocks[i](out)

        for i in range(self.depth):
            skip = encodingFeatures[-i -1]
            cmk = coordKeys[-i -1]
            
            out = self.decoding_layers[i](out, cmk)
            out = ME.cat(out, skip)
            out = self.decoding_blocks[i](out)

        out = self.output_block(out)
        
        return out
