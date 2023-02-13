import torch
import torch.nn as nn

import numpy as np

def MSE(output, truth):
    diff = (output - truth).features
    return nn.MSELoss()(diff, torch.zeros_like(diff))

def NLL_homog(output, truth):
    diff = (output - truth).features[0]
    sigma = torch.ones_like(diff)
    # u = output.features[:,1] # sigma = exp(u)

    # print ("some diffs nan!", torch.any(torch.isnan(diff)))
    # print ("some sigma zero!", torch.any(sigma == 0))
    # if torch.any(sigma == 0):
    #     print (output.features[:,1][sigma == 0])
    
    logp = -0.5*torch.pow(diff/sigma, 2) - torch.log(sigma) # + np.log(np.sqrt(2*np.pi)), ignored
    # logp = -0.5*torch.pow(diff, 2)*torch.exp(-2*u) - u # + np.log(np.sqrt(2*np.pi)), ignored

    LL = torch.sum(logp)

    # print ("LL!", LL)

    return -LL

def NLL(output, truth):
    diff = (output - truth).features[:,0]
    # sigma = torch.exp(output.features[:,1])
    sigma = torch.abs(1 + output.features[:,1])
    # u = output.features[:,1] # sigma = exp(u)

    # print ("some diffs nan!", torch.any(torch.isnan(diff)))
    # print ("some sigma zero!", torch.any(sigma == 0))
    # if torch.any(sigma == 0):
    #     print (output.features[:,1][sigma == 0])
    
    logp = -0.5*torch.pow(diff/sigma, 2) - torch.log(sigma) # + np.log(np.sqrt(2*np.pi)), ignored
    # logp = -0.5*torch.pow(diff, 2)*torch.exp(-2*u) - u # + np.log(np.sqrt(2*np.pi)), ignored

    LL = torch.sum(logp)

    # print ("LL!", LL)

    return -LL
