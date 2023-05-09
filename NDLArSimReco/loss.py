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
        truthTensor = truth.features[:,0]
        return self.loss(truth, *self.feature_map(output))

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
        sigma = torch.relu(outputSparseTensor.features[:,1]) + epsilon

        return mean, sigma
    def loss(self, truth, mean, sigma):    
        y = (mean - truth)/sigma
        
        logp = -0.5*(y + torch.exp(-y)) - torch.log(sigma) - np.log(np.sqrt(2*np.pi))

        LL = torch.sum(logp)

        return -LL
