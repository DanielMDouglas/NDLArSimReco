import MinkowskiEngine as ME
# ME.set_sparse_tensor_operation_mode(ME.SparseTensorOperationMode.SHARE_COORDINATE_MANAGER)

import torch
import torch.nn as nn
import torch.optim as optim

from torch.profiler import profile, record_function, ProfilerActivity
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import yaml

import tqdm

import os

from . import loss
from .layers import uresnet_layers, blocks
from .trainLogging import *
from .utils import sparseTensor

lossDict = {'NLL': loss.NLL,
            'NLL_moyal': loss.NLL_moyal,
            'NLL_reluError': loss.NLL_reluError,
            'NLL_reluError_masked': loss.NLL_reluError_masked,
            'MSE': loss.MSE,
            'MSE_to_scalar': loss.MSE_to_scalar,
            'NLLhomog': loss.NLL_homog,
            'voxOcc': loss.voxOcc,
            'NLL_voxOcc': loss.NLL_voxOcc,
            'NLL_voxOcc_masked': loss.NLL_voxOcc_masked,
            'NLL_voxOcc_softmax_masked': loss.NLL_voxOcc_softmax_masked,
            'MSE_voxOcc_softmax_masked': loss.MSE_voxOcc_softmax_masked,
            'MSE_voxOcc_softmax_masked_totE': loss.MSE_voxOcc_softmax_masked_totE,
            'NLL_voxOcc_softmax_masked_inference': loss.NLL_voxOcc_softmax_masked_inference,
            'CrossEntropy': loss.CrossEntropy,
            'semanticSegmentationCrossEntropy': loss.semanticSegmentationCrossEntropy,
            }

def init_layers(layerDictList, in_feat, D):
    layer_in_feat = in_feat

    for layerDict in layerDictList:

        if layerDict['type'] == 'MConvolution':
            layer_out_feat = int(layerDict['out_feat'])
            layer = ME.MinkowskiConvolution(
                in_channels = layer_in_feat,
                out_channels = layer_out_feat,
                kernel_size = int(layerDict['kernel_size']),
                stride = int(layerDict['stride']),
                bias = False,
                dimension = D,
            )
            layer_in_feat = layer_out_feat
        elif layerDict['type'] == 'MReLU':
            layer = ME.MinkowskiReLU()
        elif layerDict['type'] == 'MBatchNorm':
            layer_out_feat = layer_in_feat
            layer = ME.MinkowskiBatchNorm(layer_out_feat)
        elif layerDict['type'] == 'MMaxPooling':
            layer = ME.MinkowskiMaxPooling(
                kernel_size = int(layerDict['kernel_size']),
                stride = int(layerDict['stride']),
                dimension = D,
            )
        elif layerDict['type'] == 'MAvgPooling':
            layer = ME.MinkowskiAvgPooling(
                kernel_size = int(layerDict['kernel_size']),
                stride = int(layerDict['stride']),
                dimension = D,
            )
        elif layerDict['type'] == 'MGlobalMaxPooling':
            layer = ME.MinkowskiGlobalMaxPooling()
        elif layerDict['type'] == 'MGlobalAvgPooling':
            layer = ME.MinkowskiGlobalAvgPooling()
        elif layerDict['type'] == 'MGlobalSumPooling':
            layer = ME.MinkowskiGlobalSumPooling()
        elif layerDict['type'] == 'MLinear':
            layer_out_feat = int(layerDict['out_feat'])
            layer = ME.MinkowskiLinear(
                layer_in_feat,
                layer_out_feat,
            )
            layer_in_feat = layer_out_feat
        elif layerDict['type'] == 'MDropout':
            layer = ME.MinkowskiDropout()
        elif layerDict['type'] == 'MGlobalPooling':
            layer = ME.MinkowskiGlobalPooling()
        elif layerDict['type'] == 'UResNet':
            layer_out_feat = int(layerDict['out_feat'])
            layer = uresnet_layers.UResNet(
                layer_in_feat,
                layer_out_feat,
                int(layerDict['kernel_size']),
                int(layerDict['depth']),
            )
            layer_in_feat = layer_out_feat
        elif layerDict['type'] == 'ResNetEncoder':
            layer_out_feat = int(layerDict['out_feat'])
            layer = uresnet_layers.ResNetEncoder(
                layer_in_feat,
                int(layerDict['depth']),
            )
            layer_in_feat = layer_out_feat
        elif layerDict['type'] == 'ResNetBlock':
            layer_out_feat = int(layerDict['out_feat'])
            layer = blocks.ResNetBlock(
                layer_in_feat,
                layer_out_feat,
                kernel_size = int(layerDict['kernel_size']),
            )
            layer_in_feat = layer_out_feat
        elif layerDict['type'] == 'ResNetEncoderBlock':
            layer_out_feat = int(layerDict['out_feat'])
            if 'dropout' in layerDict:
                layer = blocks.ResNetEncoderBlock(
                    layer_in_feat,
                    layer_out_feat,
                    kernel_size = int(layerDict['kernel_size']),
                    dropout = bool(layerDict['dropout']),
                )
            else:
                layer = blocks.ResNetEncoderBlock(
                    layer_in_feat,
                    layer_out_feat,
                    kernel_size = int(layerDict['kernel_size']),
                )
            layer_in_feat = layer_out_feat
        elif layerDict['type'] == 'DownSample':
            layer_out_feat = int(layerDict['out_feat'])
            layer = blocks.DownSample(
                layer_in_feat,
                layer_out_feat,
                kernel_size = 2,
            )
            layer_in_feat = layer_out_feat
        elif layerDict['type'] == 'UResNetDropout':
            layer_out_feat = int(layerDict['out_feat'])
            layer = uresnet_layers.UResNet_dropout(
                layer_in_feat,
                layer_out_feat,
                kernel_size = int(layerDict['kernel_size']),
                depth = int(layerDict['depth']),
                dropout_depth = int(layerDict['dropout_depth']),
            )
            layer_in_feat = layer_out_feat
        elif layerDict['type'] == 'UResNetFullEncDropout':
            layer_out_feat = int(layerDict['out_feat'])
            layer = uresnet_layers.UResNet_dropout(
                layer_in_feat,
                layer_out_feat,
                kernel_size = int(layerDict['kernel_size']),
                depth = int(layerDict['depth']),
                dropout_depth = int(layerDict['dropout_depth']),
                fullEncDropout = True,
            )
            layer_in_feat = layer_out_feat
        elif layerDict['type'] == 'UNet':
            layer_out_feat = int(layerDict['out_feat'])
            layer = uresnet_layers.UNet(
                layer_in_feat,
                layer_out_feat,
                int(layerDict['depth']),
            )
            layer_in_feat = layer_out_feat
        elif layerDict['type'] == 'UNetDropout':
            layer_out_feat = int(layerDict['out_feat'])
            layer = uresnet_layers.UNet_dropout(
                layer_in_feat,
                layer_out_feat,
                int(layerDict['depth']),
            )
            layer_in_feat = layer_out_feat
        elif layerDict['type'] == 'Scaling':
            layer = blocks.Scaling(float(layerDict['scalingFactor']))
        elif layerDict['type'] == 'FeatureSelect':
            layer_out_feat = len(layerDict['featureColumns'])
            layer = blocks.FeatureSelect(layerDict['featureColumns'])
        elif layerDict['type'] == 'Threshold':
            layer = blocks.Threshold(layerDict['threshold'],
                                     layerDict['featureColumn'])
        elif layerDict['type'] == 'VoxelOccupancyHead':
            layer = blocks.VoxelOccupancyHead(layer_in_feat)
            layer_in_feat = 3
        elif layerDict['type'] == 'VoxelOccupancyHeadDeterministic':
            layer = blocks.VoxelOccupancyHeadDeterministic(layer_in_feat)
            layer_in_feat = 2
        elif layerDict['type'] == 'SemanticSegmentationHead':
            layer = blocks.SemanticSegmentationHead(layer_in_feat)
            layer_in_feat = 1

        yield layer
        
class ConfigurableSparseNetwork(ME.MinkowskiNetwork):
    def __init__(self, D, manifest, make_output = True):
        super(ConfigurableSparseNetwork, self).__init__(D)

        self.manifest = loadManifestDict(manifest)

        in_feat = self.manifest['in_feat']

        # make the output data structure
        if make_output:
            self.outDir = self.manifest['outdir']
            self.reportFile = os.path.join(self.manifest['outdir'],
                                           'train_report.dat')
            self.make_output_tree()

            self.log_manager = LogManager(self)

        self.n_epoch = 0
        self.n_iter = 0

        self.criterion = lossDict[self.manifest['loss']]()

        # load layer structure from the manifest
        self.layers = []
        for layer in init_layers(self.manifest['layers'], in_feat, D):
            self.layers.append(layer)
                
        self.network = nn.Sequential(*self.layers)
        
        if 'lr' in self.manifest:
            self.lr = self.manifest['lr']
        else:
            self.lr = 1.e-4
        self.optimizer = optim.Adam(self.parameters(), 
                                    lr = self.lr)
            
    def forward(self, x):
        return self.network(x)

    def make_checkpoint(self, filename):
        print ("saving checkpoint ", filename)
        torch.save(dict(model = self.state_dict()), filename)

        if not 'checkpoints' in self.manifest:
            self.manifest['checkpoints'] =  []
        self.manifest['checkpoints'].append(filename)

        with open(os.path.join(self.outDir, 'manifest.yaml'), 'w') as mf:
            print ('dumping network manifest to', os.path.join(self.outDir, 'manifest.yaml'))
            yaml.dump(self.manifest, mf)

    def load_checkpoint(self, filename):
        print ("loading checkpoint ", filename)
        with open(filename, 'rb') as f:
            checkpoint = torch.load(f,
                                    map_location = device)
            self.load_state_dict(checkpoint['model'], strict=False)

    def make_output_tree(self):
        # make sure the necessary directories exist
        neededDirs = [self.outDir,
                      os.path.join(self.outDir,
                                   "checkpoints"),
                      os.path.join(self.outDir,
                                   "checkpoints")]
        for thisDir in neededDirs:
            if not os.path.exists(thisDir):
                os.mkdir(thisDir)

        # update the manifest with existing checkpoints
        self.manifest['checkpoints'] = []
        for existingCheckpoint in os.listdir(os.path.join(self.outDir,
                                                          "checkpoints")):
            fullPath = os.path.join(self.outDir,
                                    "checkpoints",
                                    existingCheckpoint)
            self.manifest['checkpoints'].append(fullPath)
        self.manifest['checkpoints'].sort(key = lambda name: int(name.split('_')[-2]) + \
                                          int(name.split('_')[-1].split('.')[0])*0.001)

        # update the local copy of the manifest
        with open(os.path.join(self.outDir, 'manifest.yaml'), 'w') as mf:
            yaml.dump(self.manifest, mf)

    def MCdropout(self):
        def activate_dropout_children(network):
            for child in network.children():
                if type(child) == ME.MinkowskiDropout:
                    child.train()
                else:
                    activate_dropout_children(child)

        self.eval()
        activate_dropout_children(self)
            
    def trainLoop(self, dataLoader, dropout = False):
        """
        page through a training file, do forward calculation, evaluate loss, and backpropagate
        """

        self.train()
        
        nEpochs = int(self.manifest['nEpochs'])
       
        report = False
        prevRemainder = 0

        for i in tqdm.tqdm(range(nEpochs)):
            if i < self.n_epoch:
                print ("skipping epoch", i)
            else:
                transform = sparseTensor.transformFactory[self.manifest['transform']]()
                pbar = tqdm.tqdm(enumerate(dataLoader.load(transform = transform)),
                                 total = dataLoader.batchesPerEpoch)
                for j, (inpt, truth) in pbar:
                    if j < self.n_iter:
                        continue
                    else:
                        self.optimizer.zero_grad()

                        if report:  
                            with profile(activities=[ProfilerActivity.CUDA],
                                         profile_memory = True,
                                         record_shapes = True) as prof:
                                with record_function("model_inference"):
                                    output = self(inpt)
                                    
                            print(prof.key_averages().table(sort_by="self_cuda_time_total", 
                                                            row_limit = 10))
                    
                        else:
                            output = self.forward(inpt)

                        loss = self.criterion(output, truth)
 
                        pbarMessage = " ".join(["epoch:",
                                               str(self.n_epoch),
                                               "loss:",
                                               str(round(loss.item(), 4))])
                        pbar.set_description(pbarMessage)
                        self.training_report(loss)
                
                        # save a checkpoint of the model every 10% of an epoch
                        # remainder = (self.n_iter/dataLoader.batchesPerEpoch)%0.1
                        remainder = (self.n_iter/dataLoader.batchesPerEpoch)%1
                        if remainder < prevRemainder:
                            try:
                                self.log_manager.log_state()
                                device.empty_cache()
                            except AttributeError:
                                pass
                        prevRemainder = remainder

                        loss.backward()
                        self.optimizer.step()        

                        self.n_iter += 1
                        
                self.n_epoch += 1
                self.n_iter = 0

        self.log_manager.log_state()
        print ("final loss:", loss.item())        

    def training_report(self, loss):
        """
        Add to the running report file at a certain moment in the training process
        now controlled by logManager
        """

        self.log_manager.lossBuffer.append([self.n_epoch, 
                                            self.n_iter, 
                                            loss.item()])

    def rewind_report(self):
        """
        Load the existing report and remove every entry after the current n_epoch, n_iter
        """

        with open(self.reportFile, 'r') as rf:
            linesData = rf.readlines()

        goodLines = [line for line in linesData
                     if int(line.split('\t')[0]) < self.n_epoch]

        with open(self.reportFile, 'w') as rf:
            rf.writelines(goodLines)
        
    def evalLoop(self, dataLoader, nBatches = 50, evalMode = 'eval', accuracy = False):
        """
        page through a test file, do forward calculation, evaluate loss and accuracy metrics
        do not update the model!
        """

        evalBatches = nBatches
       
        report = False
        # report = True

        if evalMode == 'eval':
            self.eval()
        elif evalMode == 'MCdropout':
            self.MCdropout()
        elif evalMode == 'train':
            self.train()
        else:
            raise Exception ("not a valid inference mode!")
        
        lossList = []
        if accuracy:
            accList = []
        
        transform = sparseTensor.transformFactory[self.manifest['transform']](augment = False)
        pbar = tqdm.tqdm(enumerate(dataLoader.load(transform = transform)),
                         total = evalBatches)
        for i, (inpt, truth) in pbar:
            if i >= evalBatches:
                break # we're done here

            output = self.forward(inpt)
            loss = self.criterion(output, truth)

            if accuracy:
                prediction = torch.sigmoid(output.features[:,0]) > 0.5
                truth = truth.features[:,0]
                print ("truth mean", torch.mean(truth))
                thisAccuracy = (sum(prediction == truth)/len(prediction))# .cpu()
                accList.append(thisAccuracy.item())
                pbar.set_description("loss: "+str(round(loss.item(), 4)) + \
                                     " acc: "+str(round(thisAccuracy.item(), 4)))
            else:
                pbar.set_description("loss: "+str(round(loss.item(), 4)))


            lossList.append(loss.item())

        if accuracy:
            return lossList, accList
        else:
            return lossList
