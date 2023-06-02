import torch
import torch.nn as nn

import numpy as np

class loss:
    def __init__(self):
        pass
    def feature_map(self, outputSparseTensor):
        """
        Each subclass implements a function which takes
        the network output sparse tensor and returns a
        tuple of features mapped to their physical meaning.
        If not specified, just pass the 0th feature
        """
        feature = outputSparseTensor.features[:,0]
        return feature,
    def truth_map(self, truth):
        truthTensor = truth.features[:,0]
        return truthTensor
    def loss(self, truth, *mappedFeatures):
        """
        Each subclass implements a loss function which
        accepts the G.T. (sparseTensor), mapped feature tensors
        (as a tuple), and returns a rank-0 tensor
        representing the loss value
        """
        return 0
    def __call__(self, output, truth):
        """
        Compose the loss function with the feature map
        """
        return self.loss(self.truth_map(truth), *self.feature_map(output))

class MSE_stock (loss): 
    def loss(self, truth, pred):
        diff = (pred - truth)
        return nn.MSELoss()(diff, torch.zeros_like(diff))

class MSE (loss):
    def loss(self, truth, pred):
        diff = (pred - truth)

        se = torch.pow(diff, 2)
        mse = torch.sum(se)/len(diff)

        return mse

class NLL_homog (loss):
    def loss(self, truth, mean):
        sigma = torch.ones_like(mean)

        diff = (mean - truth)
    
        logp = -0.5*torch.pow(diff/sigma, 2) - torch.log(sigma)
        # divide by number of points (to help smoothness batch to batch)
        LL = torch.sum(logp)/len(diff) 
        
        return -LL
    
class NLL (loss):
    def feature_map(self, outputSparseTensor):
        mean = outputSparseTensor.features[:,0]

        epsilon = 1.e-2
        sigma = torch.exp(outputSparseTensor.features[:,1]) + epsilon

        return mean, sigma
    def loss(self, truth, mean, sigma):    
        diff = (mean - truth)
        
        logp = -0.5*torch.pow(diff/sigma, 2) - torch.log(sigma) # + np.log(np.sqrt(2*np.pi)), ignored

        LL = torch.sum(logp)/len(diff)
        
        return -LL

class NLL_reluError (loss):
    def feature_map(self, outputSparseTensor):
        mean = outputSparseTensor.features[:,0]

        epsilon = 1.e-2 
        sigma = torch.relu(outputSparseTensor.features[:,1]) + epsilon

        return mean, sigma
    def loss(self, truth, mean, sigma):    
        diff = (mean - truth)
        logp = -0.5*torch.pow(diff/sigma, 2) - torch.log(sigma) # + np.log(np.sqrt(2*np.pi)), ignored

        LL = torch.sum(logp)/len(diff)
        
        return -LL

class NLL_moyal (loss):
    def feature_map(self, outputSparseTensor):
        mean = outputSparseTensor.features[:,0]

        epsilon = 1.e-2
        # the two instances of epsilon here mean that epsilon can be adjusted during training
        # a predicted value above the previous epsilon will not change
        sigma = torch.relu(outputSparseTensor.features[:,1] - epsilon) + epsilon

        return mean, sigma
    def loss(self, truth, mean, sigma):
        y = (truth - mean)/sigma

        logp  = -1.*(y + torch.exp(-1.*y))/2 - np.log(2*np.pi)/2 - torch.log(sigma)
        
        LL = torch.sum(logp)/len(y)

        return -LL

class NLL_voxProp (loss):
    def feature_map(self, outputSparseTensor):
        mean = outputSparseTensor.features[:,0]

        epsilon = 1.e-2 
        sigma = torch.relu(outputSparseTensor.features[:,1]) + epsilon

        isFilledVoxel = outputSparseTensor.features[:,2]
        isEmptyVoxel = outputSparseTensor.features[:,3]

        return mean, sigma, isFilledVoxel, isEmptyVoxel
    def loss(self, truth, mean, sigma, isFilledVoxel, isEmptyVoxel):    
        voxelMask = torch.softmax(torch.stack([isFilledVoxel, isEmptyVoxel]), axis = 0)

        diff = (mean - truth)
        logp = -0.5*torch.pow(diff/sigma, 2) - torch.log(sigma) # + np.log(np.sqrt(2*np.pi)), ignored

        maskedLogP = torch.where(voxelMask, logp, torch.zeros_like(logp))
        
        LL = torch.sum(logp)/len(diff)
        NLL = -LL
        
        return maskedNLL

class CrossEntropy (loss):
    def feature_map(self, outputTensor):
        return outputTensor.features,

    def truth_map(self, truth):
        return truth
        
    def loss(self, truth, output):
        return nn.CrossEntropyLoss()(output, truth)
