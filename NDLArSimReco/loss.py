import torch
import torch.nn as nn

import numpy as np

def MSE_stock(output, truth):
    diff = (output - truth).features
    return nn.MSELoss()(diff, torch.zeros_like(diff))

def MSE(output, truth):
    # this version only considers the 0th feature
    diff = (output - truth).features[:, 0]
    
    se = torch.pow(diff, 2)
    mse = torch.sum(se)/len(diff)

    return mse

def NLL_homog(output, truth):

    diff = (output - truth).features[:, 0]
    sigma = torch.ones_like(diff)
    
    logp = -0.5*torch.pow(diff/sigma, 2) - torch.log(sigma)
    # divide by number of points (to help smoothness batch to batch)
    LL = torch.sum(logp)/len(diff) 

    return -LL

def NLL(output, truth):
    diff = (output - truth).features[:,0]
    sigma = torch.exp(output.features[:,1])
    # sigma = torch.abs(1 + output.features[:,1])
    
    logp = -0.5*torch.pow(diff/sigma, 2) - torch.log(sigma) # + np.log(np.sqrt(2*np.pi)), ignored

    LL = torch.sum(logp)/len(diff)

    return -LL
