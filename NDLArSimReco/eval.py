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

def plot(manifest, epoch, meanLoss, errLoss):
    plotDir = os.path.join(manifest['outdir'],
                           "plots")

    fig = plt.figure()
    ax = fig.gca()
        
    errLoss = np.abs(np.array(errLoss).T - np.array(meanLoss))
    print ("shape: ", np.array(errLoss).shape)
    ax.errorbar(epoch, meanLoss, 
                yerr = errLoss, 
                fmt = 'o')
    
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch')
    
    plt.savefig(os.path.join(plotDir,
                             'lossAcc.png'))

def save_record(manifest, epoch, meanLoss, errLoss):
    outArray = np.ndarray((4, len(epoch)))
    outArray[0,:] = epoch
    outArray[1,:] = meanLoss
    outArray[2,:] = errLoss[:,0]
    outArray[3,:] = errLoss[:,1]

    np.savetxt(os.path.join(manifest['outdir'],
                            "testEval.dat"),
               outArray)

def main(args):
    with open(args.manifest) as mf:
        manifest = yaml.load(mf, Loader = yaml.FullLoader)

    print ("initializing network...")
    net = ConfigurableSparseNetwork(in_feat=1, D=3, manifest = manifest).to(device)

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
    # dl = DataLoader(infileList, batchSize = 32)

    meanLoss = []
    errLoss = []

    epoch = []

    for e, checkpoint in enumerate(lastCheckpoints):
        net.load_checkpoint(checkpoint)
        loss = net.evaluateLoop(dl)
        print ("epoch:", e, "loss:", loss)

        epoch.append(e)

        meanLoss.append(np.mean(loss))
        errLoss.append(np.quantile(loss, (0.16, 0.84)))

    meanLoss = np.array(meanLoss)
    errLoss = np.array(errLoss)
            
    save_record(manifest, epoch, meanLoss, errLoss)
    plot(manifest, epoch, meanLoss, errLoss)
    
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
