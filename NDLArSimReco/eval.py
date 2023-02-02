import torch
# torch.manual_seed(12)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import random
# random.seed(12)

import numpy as np
# np.random.seed(12)

from NDLArSimReco.network import ConfigurableSparseNetwork
from NDLArSimReco.dataLoader import DataLoader

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import yaml
import os
            
def main(args):
    with open(args.manifest) as mf:
        manifest = yaml.load(mf, Loader = yaml.FullLoader)

    print ("initializing network...")
    net = ExampleNetwork(in_feat=1, out_feat=1, D=3, manifest = manifest).to(device)

    epochs = np.unique([int(checkpoint.split('_')[-2]) for checkpoint in manifest['checkpoints']])

    lastCheckpoints = []

    for thisEpoch in epochs:
        theseCheckpoints = []
        for checkpoint in manifest['checkpoints']:
            n_epoch = int(checkpoint.split('_')[-2])
            if thisEpoch == n_epoch:
                theseCheckpoints.append(checkpoint)

        lastCheckpoints.append(theseCheckpoints[-1])

    print ("last checkpoints: ", lastCheckpoints)

    infilePath = manifest['testfilePath'] 
    if os.path.isdir(infilePath[0]):
        infileList = [os.path.join(infilePath[0], thisFile) 
                      for thisFile in os.listdir(infilePath[0])]
        print ("loading files from list", infileList)
    else:
        infileList = infilePath
        print ("loading files from list", infileList)
    print ("initializing data loader...")
    dl = DataLoader(infileList, batchSize = manifest['batchSize'])

    meanLoss = []
    errLoss = []

    epoch = []

    for e, checkpoint in enumerate(lastCheckpoints):
        net.load_checkpoint(checkpoint)
        loss = net.evaluate(dl)

        epoch.append(e)

        meanLoss.append(np.mean(loss))
        errLoss.append(np.quantile(loss, (0.16, 0.84)))

    plotDir = os.path.join(manifest['outdir'],
                           "plots")
            
    fig = plt.figure()
    gs = GridSpec(2, 1,
                  figure = fig,
                  height_ratios = [0.5, 0.5],
                  hspace = 0)
    ax = fig.gca()
        
    errLoss = np.abs(np.array(errLoss).T - np.array(meanLoss))
    print ("shape: ", np.array(errLoss).shape)
    ax.errorbar(epoch, meanLoss, 
                yerr = errLoss, 
                fmt = 'o')
    ax.axhline(y = -np.log(1./5), 
               ls = '--') # "random guess" loss is -log(0.2)
    
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch')
    
    plt.savefig(os.path.join(plotDir,
                             'lossAcc.png'))
    
    outArray = np.ndarray((7, len(epoch)))
    outArray[0,:] = epoch
    outArray[1,:] = meanLoss
    outArray[2,:] = errLoss[0,:]
    outArray[3,:] = errLoss[1,:]

    np.savetxt(os.path.join(manifest['outdir'],
                            "testEval.dat"),
               outArray)

    
if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--manifest', type = str,
                        default = "/home/dan/studies/NDLArForwardME/testManifest.yaml",
                        help = "network manifest yaml file")
    parser.add_argument('-v', '--verbose',
                        action = 'store_true',
                        help = "print extra debug messages")
    
    
    args = parser.parse_args()
    
    main(args)
