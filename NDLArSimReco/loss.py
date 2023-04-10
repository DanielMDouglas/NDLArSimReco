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
    epsilon = 1.e-2
    sigma = torch.exp(output.features[:,1]) + epsilon
    
    logp = -0.5*torch.pow(diff/sigma, 2) - torch.log(sigma) # + np.log(np.sqrt(2*np.pi)), ignored

    LL = torch.sum(logp)/len(diff)

    return -LL

def moyal_standardized(x):
    return torch.exp(-(x + torch.exp(-x))/2)/np.sqrt(2*np.pi)

def moyalPDF(x, loc, scale):
    y = (x - loc)/scale
    return moyal_standardized(y)/scale

def NLLmoyal(output, truth):
    # epsilon = 1.e-2
    # # mean = torch.log(output.features[:,0]) + epsilon
    # mean = torch.relu(output.features[:,0]) + epsilon
    # sigma = torch.exp(output.features[:,1]) + epsilon

    meanLL = 0.
    meanUL = 5.
    meanDR = meanUL - meanLL
    mean = meanDR*torch.sigmoid(output.features[:,0]) + meanLL

    sigmaLL = 1.e-2
    sigmaUL = 3.
    sigmaDR = sigmaUL - sigmaLL
    sigma = sigmaDR*torch.sigmoid(output.features[:, 1]) + sigmaLL

    obs = truth.features[:,0]
    
    y = (obs - mean)/sigma

    print (truth.shape)
    print ("obs range", torch.min(obs).item(), torch.max(obs).item())

    # print (mean)
    print (torch.mean(output.features[:,0]).item())
    print ("mean range", torch.min(mean).item(), torch.max(mean).item())
    # print (torch.any(torch.isnan(mean)))
    # print (torch.all(mean > 0))

    # print (sigma)
    print (torch.mean(output.features[:,1]).item())
    print ("sigma range", torch.min(sigma).item(), torch.max(sigma).item())
    # print (torch.any(torch.isnan(sigma)))
    # print (torch.all(sigma > 0))

    print ("y range", torch.min(y).item(), torch.max(y).item()) 
    print (y)
    print (torch.exp(-y))
    print (torch.any(torch.isinf(torch.exp(-y))))
    print (torch.log(sigma))
    print (torch.any(torch.isinf(torch.log(sigma))))
    
    logp = -0.5*(y + torch.exp(-y)) - torch.log(sigma) - np.log(np.sqrt(2*np.pi))

    LL = torch.sum(logp)

    return -LL

def NLLeval(output, truth):
    diff = torch.relu((output - truth).features[:,0])
    epsilon = 1.e-2
    sigma = torch.exp(output.features[:,1]) + epsilon
    
    logp = -0.5*torch.pow(diff/sigma, 2) - torch.log(sigma) + np.log(np.sqrt(2*np.pi))

    LL = torch.sum(logp)/len(diff)

    return -LL

def NLL_reluError(output, truth):
    diff = (output - truth).features[:,0]
    epsilon = 1.e-2
    sigma = torch.relu(output.features[:,1]) + epsilon
    
    logp = -0.5*torch.pow(diff/sigma, 2) - torch.log(sigma) # + np.log(np.sqrt(2*np.pi)), ignored

    LL = torch.sum(logp)/len(diff)

    return -LL

def NLLr(output, truth):
    return NLL(output, truth)-NLL(output, output)
