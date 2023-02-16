import MinkowskiEngine as ME
ME.set_sparse_tensor_operation_mode(ME.SparseTensorOperationMode.SHARE_COORDINATE_MANAGER)

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

import ot

from . import loss

lossDict = {'NLL': loss.NLL,
            'NLL_reluError': loss.NLL_reluError,
            'MSE': loss.MSE,
            'NLLhomog': loss.NLL_homog,
}

class ConfigurableSparseNetwork(ME.MinkowskiNetwork):
    def __init__(self, in_feat, D, manifest):
        super(ConfigurableSparseNetwork, self).__init__(D)

        # save the manifest dict internally
        self.manifest = manifest

        # make the output data structure
        self.outDir = self.manifest['outdir']
        self.reportFile = os.path.join(self.manifest['outdir'],
                                       'train_report.dat')
        self.make_output_tree()

        self.n_epoch = 0
        self.n_iter = 0

        if 'lr' in self.manifest:
            self.lr = self.manifest['lr']
        else:
            self.lr = 1.e-4

        self.criterion = lossDict[self.manifest['loss']] 

        # load layer structure from the manifest
        self.layers = []
        layer_in_feat = in_feat
        for layer in self.manifest['layers']:
            if layer['type'] == 'MConvolution':
                layer_out_feat = int(layer['out_feat'])
                self.layers.append(ME.MinkowskiConvolution(
                    in_channels = layer_in_feat,
                    out_channels = layer_out_feat,
                    kernel_size = int(layer['kernel_size']),
                    stride = int(layer['stride']),
                    bias = False,
                    dimension = D
                ))
                layer_in_feat = layer_out_feat
            elif layer['type'] == 'MReLU':
                self.layers.append(ME.MinkowskiReLU())
            elif layer['type'] == 'MBatchNorm':
                layer_out_feat = layer_in_feat
                self.layers.append(ME.MinkowskiBatchNorm(layer_out_feat))
            elif layer['type'] == 'MMaxPooling':
                self.layers.append(ME.MinkowskiMaxPooling(
                    kernel_size = int(layer['kernel_size']),
                    stride = int(layer['stride']),
                    dimension = D
                ))
            elif layer['type'] == 'MLinear':
                layer_out_feat = int(layer['out_feat'])
                self.layers.append(ME.MinkowskiLinear(
                    layer_in_feat,
                    layer_out_feat
                ))
                layer_in_feat = layer_out_feat
            elif layer['type'] == 'MGlobalPooling':
                self.layers.append(ME.MinkowskiGlobalPooling())

        self.network = nn.Sequential(*self.layers)
            
    def forward(self, x):
        return self.network(x)

    def make_checkpoint(self, filename):
        print ("saving checkpoint ", filename)
        torch.save(dict(model = self.state_dict()), filename)

        if not 'checkpoints' in self.manifest:
            self.manifest['checkpoints'] =  []
        self.manifest['checkpoints'].append(filename)

        with open(os.path.join(self.outDir, 'manifest.yaml'), 'w') as mf:
            print ('dumping manifest to', os.path.join(self.outDir, 'manifest.yaml'))
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
            
    def train(self, dataLoader):
        """
        page through a training file, do forward calculation, evaluate loss, and backpropagate
        """
        optimizer = optim.SGD(self.parameters(), 
                              lr = self.lr, 
                              momentum = 0.9)

        nEpochs = int(self.manifest['nEpochs'])
       
        report = False
        prevRemainder = 0

        # if there's a previous checkpoint, start there
        if 'checkpoints' in self.manifest and self.manifest['checkpoints'] != []:
            latestCheckpoint = self.manifest['checkpoints'][-1]
            self.load_checkpoint(latestCheckpoint)
            self.n_epoch = int(latestCheckpoint.split('_')[-2])
            self.n_iter = int(latestCheckpoint.split('_')[-1].split('.')[0])
            print ("resuming training at epoch {}, iteration {}".format(self.n_epoch, self.n_iter))

        for i in tqdm.tqdm(range(nEpochs)):
            if i < self.n_epoch:
                continue

            dataLoader.setFileLoadOrder()
            for j, (hits, edep) in tqdm.tqdm(enumerate(dataLoader.load()),
                                             total = dataLoader.batchesPerEpoch):
                if j < self.n_iter:
                    continue

                optimizer.zero_grad()

                if report:
                    with profile(activities=[ProfilerActivity.CUDA],
                                 profile_memory = True,
                                 record_shapes = True) as prof:
                        with record_function("model_inference"):
                            output = self(hits)

                    print(prof.key_averages().table(sort_by="self_cuda_time_total", 
                                                    row_limit = 10))
                    
                else:
                    output = self(hits)
                    
                loss = self.criterion(output, edep)
                print ("epoch:", self.n_epoch, 
                       "iter:", self.n_iter, 
                       "loss:", loss.item(),
                       end = '\r')
                self.training_report(loss)
                
                loss.backward()
                optimizer.step()
        
                self.n_iter += 1

                # save a checkpoint of the model every 10% of an epoch
                remainder = (self.n_iter/dataLoader.batchesPerEpoch)%0.1
                if remainder < prevRemainder:
                    try:
                        checkpointFile = os.path.join(self.outDir,
                                                      'checkpoints',
                                                      'checkpoint_'+str(self.n_epoch)+'_'+str(self.n_iter)+'.ckpt')
                        self.make_checkpoint(checkpointFile)

                        # self.training_report(loss)

                        device.empty_cache()
                    except AttributeError:
                        pass
                prevRemainder = remainder
            
            self.n_epoch += 1
            self.n_iter = 0

        print ("final loss:", loss.item())        

    def training_report(self, loss):
        """
        Add to the running report file at a certain moment in the training process
        """

        with open(self.reportFile, 'a') as rf:
            rf.write('{} \t {} \t {} \n'.format(self.n_epoch, 
                                                self.n_iter, 
                                                loss))
        
    def evaluate(self, dataLoader):
        """
        page through a test file, do forward calculation, evaluate loss and accuracy metrics
        do not update the model!
        """

        evalBatches = 50
        # evalBatches = 10
       
        report = False
        # report = True
        
        dataLoader.setFileLoadOrder()

        lossList = []
        for i, (larpix, edep) in tqdm.tqdm(enumerate(dataLoader.load()),
                                           total = evalBatches):
            if i >= evalBatches:
                break # we're done here

            if report:
                with profile(activities=[ProfilerActivity.CUDA],
                             profile_memory = True,
                             record_shapes = True) as prof:
                    with record_function("model_inference"):
                        prediction = self(larpix)

                print(prof.key_averages().table(sort_by="self_cuda_time_total", 
                                                row_limit = 10))

            else:
                prediction = self(larpix)

            loss = self.criterion(prediction, edep)
            # print ("loss", loss.item())
            
            lossList.append(loss.item())

        return lossList

