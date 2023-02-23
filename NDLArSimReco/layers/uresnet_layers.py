import torch
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import MinkowskiEngine as ME
ME.set_sparse_tensor_operation_mode(ME.SparseTensorOperationMode.SHARE_COORDINATE_MANAGER)

from NDLArSimReco.layers.blocks import ResNetBlock, CascadeDilationBlock, ASPP
from NDLArSimReco.layers.activation_normalization_factories import activations_construct
from NDLArSimReco.layers.activation_normalization_factories import normalizations_construct
from NDLArSimReco.layers.configuration import setup_cnn_configuration


class UResNetEncoder(torch.nn.Module):
    '''
    Vanilla UResNet with access to intermediate feature planes.

    Configuration
    -------------
    data_dim: int, default 3
    num_input: int, default 1
    allow_bias: bool, default False
    spatial_size: int, default 512
    leakiness: float, default 0.33
    activation: dict
        For activation function, defaults to `{'name': 'lrelu', 'args': {}}`
    norm_layer: dict
        For normalization function, defaults to `{'name': 'batch_norm', 'args': {}}`

    depth : int, default 5
        Depth of UResNet, also corresponds to how many times we down/upsample.
    filters : int, default 16
        Number of filters in the first convolution of UResNet.
        Will increase linearly with depth.
    reps : int, default 2
        Convolution block repetition factor
    input_kernel : int, default 3
        Receptive field size for very first convolution after input layer.

    Output
    ------
    encoderTensors: list of ME.SparseTensor
        list of intermediate tensors (taken between encoding block and convolution)
        from encoder half
    finalTensor: ME.SparseTensor
        feature tensor at deepest layer
    features_ppn: list of ME.SparseTensor
        list of intermediate tensors (right after encoding block + convolution)
    '''
    def __init__(self, cfg, name='uresnet_encoder'):
        # To allow UResNet to inherit directly from UResNetEncoder
        super(UResNetEncoder, self).__init__()
        #torch.nn.Module.__init__(self)
        setup_cnn_configuration(self, cfg, name)

        model_cfg = cfg.get(name, {})
        # UResNet Configurations
        self.reps = model_cfg.get('reps', 2)
        self.depth = model_cfg.get('depth', 5)
        self.num_filters = model_cfg.get('filters', 16)
        self.nPlanes = [i * self.num_filters for i in range(1, self.depth+1)]
        # self.kernel_size = cfg.get('kernel_size', 3)
        # self.downsample = cfg.get(downsample, 2)
        self.input_kernel = model_cfg.get('input_kernel', 3)

        # Initialize Input Layer
        # print(self.num_input)
        # print(self.input_kernel)
        self.input_layer = ME.MinkowskiConvolution(
            in_channels=self.num_input,
            out_channels=self.num_filters,
            kernel_size=self.input_kernel, stride=1, dimension=self.D,
            bias=self.allow_bias)

        # Initialize Encoder
        self.encoding_conv = []
        self.encoding_block = []
        for i, F in enumerate(self.nPlanes):
            m = []
            for _ in range(self.reps):
                m.append(ResNetBlock(F, F,
                    dimension=self.D,
                    activation=self.activation_name,
                    activation_args=self.activation_args,
                    normalization=self.norm,
                    normalization_args=self.norm_args,
                    bias=self.allow_bias))
            m = nn.Sequential(*m)
            self.encoding_block.append(m)
            m = []
            if i < self.depth-1:
                m.append(normalizations_construct(self.norm, F, **self.norm_args))
                m.append(activations_construct(
                    self.activation_name, **self.activation_args))
                m.append(ME.MinkowskiConvolution(
                    in_channels=self.nPlanes[i],
                    out_channels=self.nPlanes[i+1],
                    kernel_size=2, stride=2, dimension=self.D,
                    bias=self.allow_bias))
            m = nn.Sequential(*m)
            self.encoding_conv.append(m)
        self.encoding_conv = nn.Sequential(*self.encoding_conv)
        self.encoding_block = nn.Sequential(*self.encoding_block)


    def encoder(self, x):
        '''
        Vanilla UResNet Encoder.

        Parameters
        ----------
        x : MinkowskiEngine SparseTensor

        Returns
        -------
        dict
        '''
        # print('input' , self.input_layer)
        # for name, param in self.input_layer.named_parameters():
        #     print(name, param.shape, param)
        x = self.input_layer(x)
        encoderTensors = [x]
        encoderCMK = [x.coordinate_map_key]
        features_ppn = [x]
        for i, layer in enumerate(self.encoding_block):
            # print ("encoding layer", i)
            # print ("tensor", x)
            x = self.encoding_block[i](x)
            encoderTensors.append(x)
            encoderCMK.append(x.coordinate_map_key)
            x = self.encoding_conv[i](x)
            features_ppn.append(x)

        result = {
            "encoderTensors": encoderTensors,
            "encoderCMK": encoderCMK,
            "features_ppn": features_ppn,
            "finalTensor": x
        }
        return result


    def forward(self, input):
        # coords = input[:, 0:self.D+1].int()
        # features = input[:, self.D+1:].float()
        #
        # x = ME.SparseTensor(features, coordinates=coords)
        encoderOutput = self.encoder(input)
        encoderTensors = encoderOutput['encoderTensors']
        encoderCMK = encoderOutput['encoderCMK']
        finalTensor = encoderOutput['finalTensor']
        # decoderTensors = self.decoder(finalTensor, encoderTensors)

        res = {
            'encoderTensors': encoderTensors,
            'encoderCMK': encoderCMK,
            # 'decoderTensors': decoderTensors,
            'finalTensor': finalTensor,
            'features_ppn': encoderOutput['features_ppn']
        }
        return res


class UResNetDecoder(torch.nn.Module):
    """
    Vanilla UResNet Decoder

    Configuration
    -------------
    data_dim: int, default 3
    num_input: int, default 1
    allow_bias: bool, default False
    spatial_size: int, default 512
    leakiness: float, default 0.33
    activation: dict
        For activation function, defaults to `{'name': 'lrelu', 'args': {}}`
    norm_layer: dict
        For normalization function, defaults to `{'name': 'batch_norm', 'args': {}}`

    depth : int, default 5
        Depth of UResNet, also corresponds to how many times we down/upsample.
    filters : int, default 16
        Number of filters in the first convolution of UResNet.
        Will increase linearly with depth.
    reps : int, default 2
        Convolution block repetition factor

    Output
    ------
    list of ME.SparseTensor
    """
    def __init__(self, cfg, name='uresnet_decoder'):
        super(UResNetDecoder, self).__init__()
        setup_cnn_configuration(self, cfg, name)

        # UResNet Configurations
        self.model_config = cfg.get(name, {})
        self.reps = self.model_config.get('reps', 2)  # Conv block repetition factor
        #self.kernel_size = self.model_config.get('kernel_size', 2)
        self.depth = self.model_config.get('depth', 5)
        self.num_filters = self.model_config.get('filters', 16)
        self.nPlanes = [i*self.num_filters for i in range(1, self.depth+1)]
        #self.downsample = [self.kernel_size, 2]  # [filter size, filter stride]

        # self.encoder_num_filters = self.model_config.get('encoder_num_filters', None)
        # if self.encoder_num_filters is None:
        #     self.encoder_num_filters = self.num_filters
        # self.encoder_nPlanes = [i*self.encoder_num_filters for i in range(1, self.depth+1)]
        # self.nPlanes[-1] = self.encoder_nPlanes[-1]

        # Initialize Decoder
        self.decoding_block = []
        self.decoding_conv = []
        for i in range(self.depth-2, -1, -1):
            # m = []
            # m.append(normalizations_construct(self.norm, self.nPlanes[i+1], **self.norm_args))
            # m.append(activations_construct(
            #     self.activation_name, **self.activation_args))
            # m.append(ME.MinkowskiConvolutionTranspose(
            #     in_channels=self.nPlanes[i+1],
            #     out_channels=self.nPlanes[i],
            #     kernel_size=2,
            #     stride=2,
            #     dimension=self.D,
            #     bias=self.allow_bias))
            # m = nn.Sequential(*m)
            self.decoding_conv.append(ME.MinkowskiConvolutionTranspose(
                in_channels=self.nPlanes[i+1],
                out_channels=self.nPlanes[i],
                kernel_size=2,
                stride=2,
                dimension=self.D,
                bias=self.allow_bias)
            )
            m = []
            for j in range(self.reps):
                m.append(ResNetBlock(self.nPlanes[i] * (2 if j == 0 else 1),
                                     self.nPlanes[i],
                                     dimension=self.D,
                                     activation=self.activation_name,
                                     activation_args=self.activation_args,
                                     normalization=self.norm,
                                     normalization_args=self.norm_args,
                                     bias=self.allow_bias))
            m = nn.Sequential(*m)
            self.decoding_block.append(m)
        self.decoding_block = nn.Sequential(*self.decoding_block)
        self.decoding_conv = nn.Sequential(*self.decoding_conv)


    def decoder(self, final, encoderTensors, encoderCMK):
        '''
        Vanilla UResNet Decoder

        Parameters
        ----------
        encoderTensors : list of SparseTensor
            output of encoder.

        Returns
        -------
        decoderTensors : list of SparseTensor
            list of feature tensors in decoding path at each spatial resolution.
        '''
        decoderTensors = []
        x = final
        for i, layer in enumerate(self.decoding_conv):
            # eTensor = encoderTensors[-i-2]
            eTensor = encoderTensors[-i-2]
            eCMK = encoderCMK[-i-2]
            x = layer(x, eCMK)
            # print ('decoder layer', i)
            # print ("encoder layer output", eTensor)
            # print ("decoder layer input", x)
            x = ME.cat(eTensor, x)
            x = self.decoding_block[i](x)
            decoderTensors.append(x)
        return decoderTensors

    def forward(self, final, encoderTensors, encoderCMK):
        return self.decoder(final, encoderTensors, encoderCMK)


class UResNet(torch.nn.Module):
    '''
    Vanilla UResNet with access to intermediate feature planes.

    Configuration
    -------------
    data_dim: int, default 3
    num_input: int, default 1
    allow_bias: bool, default False
    spatial_size: int, default 512
    leakiness: float, default 0.33
    activation: dict
        For activation function, defaults to `{'name': 'lrelu', 'args': {}}`
    norm_layer: dict
        For normalization function, defaults to `{'name': 'batch_norm', 'args': {}}`

    depth : int, default 5
        Depth of UResNet, also corresponds to how many times we down/upsample.
    filters : int, default 16
        Number of filters in the first convolution of UResNet.
        Will increase linearly with depth.
    reps : int, default 2
        Convolution block repetition factor
    input_kernel : int, default 3
        Receptive field size for very first convolution after input layer.

    Output
    ------
    encoderTensors: list of ME.SparseTensor
        list of intermediate tensors (taken between encoding block and convolution)
        from encoder half
    decoderTensors: list of ME.SparseTensor
        list of intermediate tensors (taken between encoding block and convolution)
        from decoder half
    finalTensor: ME.SparseTensor
        feature tensor at deepest layer
    features_ppn: list of ME.SparseTensor
        list of intermediate tensors (right after encoding block + convolution)
    '''
    def __init__(self, cfg, name='uresnet'):
        super(UResNet, self).__init__()
        #UResNetEncoder.__init__(self, cfg, name=name)
        #UResNetDecoder.__init__(self, cfg, name=name)
        setup_cnn_configuration(self, cfg, name)
        self.encoder = UResNetEncoder(cfg, name=name)
        self.decoder = UResNetDecoder(cfg, name=name)

        self.num_filters = self.encoder.num_filters

        # print('Total Number of Trainable Parameters (mink/layers/uresnet) = {}'.format(
        #             sum(p.numel() for p in self.parameters() if p.requires_grad)))

    def forward(self, input):
        encoderOutput = self.encoder(input)
        encoderTensors = encoderOutput['encoderTensors']
        encoderCMK = encoderOutput['encoderCMK']
        finalTensor = encoderOutput['finalTensor']
        decoderTensors = self.decoder(finalTensor, encoderTensors, encoderCMK)

        res = {
            'encoderTensors': encoderTensors,
            'decoderTensors': decoderTensors,
            'finalTensor': finalTensor,
            'features_ppn': encoderOutput['features_ppn']
        }
        print (res['finalTensor'].shape)
        return res['finalTensor']


class presetUResNet(torch.nn.Module):
    def __init__(self, in_features, out_features, depth = 2, nFilters = 16, name='preseturesnet'):
        super(presetUResNet, self).__init__()

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
           # ME.MinkowskiReLU(),
            # ME.MinkowskiConvolution(
            #     in_channels = self.nFilters,
            #     out_channels = self.nFilters,
            #     kernel_size = 3,
            #     stride = 1,
            #     dimension = 3,
            # ),
            # ME.MinkowskiReLU(),
            # ME.MinkowskiConvolution(
            #     in_channels = self.nFilters,
            #     out_channels = self.nFilters,
            #     kernel_size = 3,
            #     stride = 1,
            #     dimension = 3,
            # ),
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
                    dimension = 3).to(device)
            )
            self.encoding_blocks.append(
                nn.Sequential(
                    ME.MinkowskiConvolution(
                        in_channels = self.featureSizesEnc[i][1],
                        out_channels = self.featureSizesEnc[i][1],
                        kernel_size = 3,
                        stride = 1,
                        dimension = 3),
                    ME.MinkowskiReLU(),
                    ME.MinkowskiConvolution(
                        in_channels = self.featureSizesEnc[i][1],
                        out_channels = self.featureSizesEnc[i][1],
                        kernel_size = 3,
                        stride = 1,
                        dimension = 3),
                    # ME.MinkowskiReLU(),
                    # ME.MinkowskiConvolution(
                    #     in_channels = self.featureSizesEnc[i][1],
                    #     out_channels = self.featureSizesEnc[i][1],
                    #     kernel_size = 3,
                    #     stride = 1,
                    #     dimension = 3),
                ).to(device)
            )

        for i in range(self.depth):
            self.decoding_layers.append(
                ME.MinkowskiConvolutionTranspose(
                    in_channels = self.featureSizesDec[i][0],
                    out_channels = self.featureSizesDec[i][1],
                    kernel_size = 2,
                    stride = 2,
                    dimension = 3).to(device)
            )
            self.decoding_blocks.append(
                nn.Sequential(
                    # ME.MinkowskiConvolution(
                    #     in_channels = 2*self.featureSizesDec[i][1],
                    #     out_channels = 2*self.featureSizesDec[i][1],
                    #     kernel_size = 3,
                    #     stride = 1,
                    #     dimension = 3),
                    # ME.MinkowskiReLU(),
                    ME.MinkowskiConvolution(
                        in_channels = 2*self.featureSizesDec[i][1],
                        out_channels = 2*self.featureSizesDec[i][1],
                        kernel_size = 3,
                        stride = 1,
                        dimension = 3),
                    ME.MinkowskiReLU(),
                    ME.MinkowskiConvolution(
                        in_channels = 2*self.featureSizesDec[i][1],
                        out_channels = self.featureSizesDec[i][1],
                        kernel_size = 3,
                        stride = 1,
                        dimension = 3),
                ).to(device)
            )
        # self.decoding_layers.reverse()

        self.output_block = nn.Sequential(
            # ME.MinkowskiConvolution(
            #     in_channels = self.featureSizesDec[-1][1],
            #     out_channels = self.featureSizesDec[-1][1],
            #     kernel_size = 3,
            #     stride = 1,
            #     dimension = 3,
            # ),
            # ME.MinkowskiReLU(),
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
        
    def forward(self, input):
        encodingFeatures = []
        coordKeys = []

        input = self.input_block(input)
        for i in range(self.depth):
            encodingFeatures.append(input)
            coordKeys.append(input.coordinate_map_key)

            input = self.encoding_layers[i](input)
            input = self.encoding_blocks[i](input)

        for i in range(self.depth):
            skip = encodingFeatures[-i -1]
            cmk = coordKeys[-i -1]
            
            input = self.decoding_layers[i](input, cmk)
            input = ME.cat(input, skip)
            input = self.decoding_blocks[i](input)

        input = self.output_block(input)
        
        return input


class presetUResNetWithDropout(torch.nn.Module):
    def __init__(self, in_features, out_features, depth = 2, nFilters = 16, name='preseturesnet'):
        super(presetUResNetWithDropout, self).__init__()

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
           # ME.MinkowskiReLU(),
            # ME.MinkowskiConvolution(
            #     in_channels = self.nFilters,
            #     out_channels = self.nFilters,
            #     kernel_size = 3,
            #     stride = 1,
            #     dimension = 3,
            # ),
            # ME.MinkowskiReLU(),
            # ME.MinkowskiConvolution(
            #     in_channels = self.nFilters,
            #     out_channels = self.nFilters,
            #     kernel_size = 3,
            #     stride = 1,
            #     dimension = 3,
            # ),
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
                    # ME.MinkowskiReLU(),
                    # ME.MinkowskiConvolution(
                    #     in_channels = self.featureSizesEnc[i][1],
                    #     out_channels = self.featureSizesEnc[i][1],
                    #     kernel_size = 3,
                    #     stride = 1,
                    #     dimension = 3),
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
                    dimension = 3).to(device)
            )
            self.decoding_blocks.append(
                nn.Sequential(
                    # ME.MinkowskiConvolution(
                    #     in_channels = 2*self.featureSizesDec[i][1],
                    #     out_channels = 2*self.featureSizesDec[i][1],
                    #     kernel_size = 3,
                    #     stride = 1,
                    #     dimension = 3),
                    # ME.MinkowskiReLU(),
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
                ).to(device)
            )
        self.decoding_layers = nn.Sequential(*self.decoding_layers)
        self.decoding_blocks = nn.Sequential(*self.decoding_blocks)

        # self.decoding_layers.reverse()

        self.output_block = nn.Sequential(
            # ME.MinkowskiConvolution(
            #     in_channels = self.featureSizesDec[-1][1],
            #     out_channels = self.featureSizesDec[-1][1],
            #     kernel_size = 3,
            #     stride = 1,
            #     dimension = 3,
            # ),
            # ME.MinkowskiReLU(),
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
        
    def forward(self, input):
        encodingFeatures = []
        coordKeys = []

        input = self.input_block(input)
        for i in range(self.depth):
            encodingFeatures.append(input)
            coordKeys.append(input.coordinate_map_key)

            input = self.encoding_layers[i](input)
            input = self.encoding_blocks[i](input)

        for i in range(self.depth):
            skip = encodingFeatures[-i -1]
            cmk = coordKeys[-i -1]
            
            input = self.decoding_layers[i](input, cmk)
            input = ME.cat(input, skip)
            input = self.decoding_blocks[i](input)

        input = self.output_block(input)
        
        return input

class presetUResNetWithDropoutNoNonLin(torch.nn.Module):
    def __init__(self, in_features, out_features, depth = 2, nFilters = 16, name='preseturesnet'):
        super(presetUResNetWithDropout, self).__init__()

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
           # ME.MinkowskiReLU(),
            # ME.MinkowskiConvolution(
            #     in_channels = self.nFilters,
            #     out_channels = self.nFilters,
            #     kernel_size = 3,
            #     stride = 1,
            #     dimension = 3,
            # ),
            # ME.MinkowskiReLU(),
            # ME.MinkowskiConvolution(
            #     in_channels = self.nFilters,
            #     out_channels = self.nFilters,
            #     kernel_size = 3,
            #     stride = 1,
            #     dimension = 3,
            # ),
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
                    dimension = 3).to(device)
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
                    # ME.MinkowskiBatchNorm(self.featureSizesEnc[i][1]),
                    # ME.MinkowskiReLU(),
                    # ME.MinkowskiConvolution(
                    #     in_channels = self.featureSizesEnc[i][1],
                    #     out_channels = self.featureSizesEnc[i][1],
                    #     kernel_size = 3,
                    #     stride = 1,
                    #     dimension = 3),
                    # ME.MinkowskiBatchNorm(self.featureSizesEnc[i][1]),
                    # ME.MinkowskiReLU(),
                    # ME.MinkowskiReLU(),
                    # ME.MinkowskiConvolution(
                    #     in_channels = self.featureSizesEnc[i][1],
                    #     out_channels = self.featureSizesEnc[i][1],
                    #     kernel_size = 3,
                    #     stride = 1,
                    #     dimension = 3),
                ).to(device)
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
                    dimension = 3).to(device)
            )
            self.decoding_blocks.append(
                nn.Sequential(
                    # ME.MinkowskiConvolution(
                    #     in_channels = 2*self.featureSizesDec[i][1],
                    #     out_channels = 2*self.featureSizesDec[i][1],
                    #     kernel_size = 3,
                    #     stride = 1,
                    #     dimension = 3),
                    # ME.MinkowskiReLU(),
                    # ME.MinkowskiConvolution(
                    #     in_channels = 2*self.featureSizesDec[i][1],
                    #     out_channels = 2*self.featureSizesDec[i][1],
                    #     kernel_size = 3,
                    #     stride = 1,
                    #     dimension = 3),
                    # ME.MinkowskiBatchNorm(2*self.featureSizesDec[i][1]),
                    # ME.MinkowskiReLU(),
                    ME.MinkowskiConvolution(
                        in_channels = 2*self.featureSizesDec[i][1],
                        out_channels = self.featureSizesDec[i][1],
                        kernel_size = 3,
                        stride = 1,
                        dimension = 3),
                    # ME.MinkowskiBatchNorm(self.featureSizesDec[i][1]),
                    # ME.MinkowskiReLU(),
                    ME.MinkowskiDropout(),
                ).to(device)
            )
        # self.decoding_layers.reverse()
        self.decoding_layers = nn.Sequential(*self.decoding_layers)
        self.decoding_blocks = nn.Sequential(*self.decoding_blocks)
        
        self.output_block = nn.Sequential(
            # ME.MinkowskiConvolution(
            #     in_channels = self.featureSizesDec[-1][1],
            #     out_channels = self.featureSizesDec[-1][1],
            #     kernel_size = 3,
            #     stride = 1,
            #     dimension = 3,
            # ),
            # ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(
                in_channels = self.featureSizesDec[-1][1],
                out_channels = self.featureSizesDec[-1][1],
                kernel_size = 3,
                stride = 1,
                dimension = 3,
            ),
            # ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(
                in_channels = self.featureSizesDec[-1][1],
                out_channels = self.out_features,
                kernel_size = 3,
                stride = 1,
                dimension = 3,
            ),
        )
        
    def forward(self, input):
        encodingFeatures = []
        coordKeys = []

        input = self.input_block(input)
        for i in range(self.depth):
            encodingFeatures.append(input)
            coordKeys.append(input.coordinate_map_key)

            input = self.encoding_layers[i](input)
            input = self.encoding_blocks[i](input)

        for i in range(self.depth):
            skip = encodingFeatures[-i -1]
            cmk = coordKeys[-i -1]
            
            input = self.decoding_layers[i](input, cmk)
            input = ME.cat(input, skip)
            input = self.decoding_blocks[i](input)

        input = self.output_block(input)
        
        return input
