import torch
# torch.manual_seed(12)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import random
# random.seed(12)

import numpy as np
# np.random.seed(12)

from NDLArSimReco.network import ConfigurableSparseNetwork
from NDLArSimReco.dataLoader import dataLoaderFactory

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import yaml
import os

def save_record(manifest, checkpointDict):
    epoch = np.array(list(checkpointDict.keys()))
    meanLoss = np.array([thisValue['medianLoss']
                         for thisValue in checkpointDict.values()])
    errLoss = np.array([thisValue['lossInterval']
                        for thisValue in checkpointDict.values()])
    
    outArray = np.ndarray((4, len(epoch)))
    outArray[0,:] = epoch
    outArray[1,:] = meanLoss
    outArray[2,:] = errLoss[:,0]
    outArray[3,:] = errLoss[:,1]

    np.savetxt(os.path.join(manifest['outdir'],
                            "testEval.dat"),
               outArray)

def main_old(args):
    with open(args.manifest) as mf:
        manifest = yaml.load(mf, Loader = yaml.FullLoader)

    print ("initializing network...")
    net = ConfigurableSparseNetwork(D=3, manifest = manifest).to(device)

    epochs = np.unique([int(checkpoint.split('_')[-2]) for checkpoint in manifest['checkpoints']])

    firstCheckpoints = {}
    lastCheckpoints = {}

    for thisEpoch in epochs:
        theseCheckpoints = []
        for checkpoint in manifest['checkpoints']:
            n_epoch = int(checkpoint.split('_')[-2])
            n_iter = int(checkpoint.split('_')[-1].split('.')[0])
            if thisEpoch == n_epoch:
                theseCheckpoints.append(checkpoint)

        lastCheckpoints[thisEpoch+1] = {'checkpoint': theseCheckpoints[-1],
                                        'iter': n_iter}
        if thisEpoch > 0:
            firstCheckpoints[thisEpoch] = {'checkpoint': theseCheckpoints[0],
                                           'iter': n_iter}

    finalCheckpoint = os.path.join(manifest['outdir'],
                                   'checkpoint_final_'+str(manifest['nEpochs'])+'_0.ckpt') 
    if os.path.exists(finalCheckpoint):
        firstCheckpoints[thisEpoch+1] = {'checkpoint': finalCheckpoint,
                                         'iter': 0}

    if args.useLast:
        theseCheckpoints = lastCheckpoints
    else:
        theseCheckpoints = firstCheckpoints

    print ("using found checkpoints", theseCheckpoints)

    infilePath = manifest['testfilePath'] 
    # infilePath = manifest['trainfilePath'] 
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

    for epoch, valDict in theseCheckpoints.items():
        checkpoint = valDict['checkpoint']
        net.load_checkpoint(checkpoint)
        loss = net.evalLoop(dl,
                            nBatches = args.nBatches,
                            evalMode = not args.trainMode)
        
        # meanLoss.append(np.mean(loss))
        medianLoss = np.median(loss)
        lossInterval = np.quantile(loss, (0.16, 0.84))

        valDict['medianLoss'] = medianLoss
        valDict['lossInterval'] = lossInterval

    save_record(manifest, theseCheckpoints)
def main(args):
    with open(args.manifest) as mf:
        manifest = yaml.load(mf, Loader = yaml.FullLoader)

    print ("initializing network...")
    net = ConfigurableSparseNetwork(D=3, manifest = manifest).to(device)

    infilePath = manifest['testfilePath'] 
    if os.path.isdir(infilePath[0]):
        infileList = [os.path.join(infilePath[0], thisFile) 
                      for thisFile in os.listdir(infilePath[0])]
        print ("loading files from list", infileList)
    else:
        infileList = infilePath
        print ("loading files from list", infileList)

    print ("initializing data loader...")
    dl = dataLoaderFactory[manifest['dataLoader']](infileList,
                                                   batchSize = manifest['batchSize'])

    for log_entry in net.log_manager.entries:
        print ("reverting from log entry:", log_entry.outDir)
        log_entry.load()
        # net.load_checkpoint(log_entry.manifest['checkpointPath'])
        print ("current model state:", net.state_dict()['network.1.encoding_blocks.0.0.conv1.kernel']) 

        loss = net.evalLoop(dl, args.nBatches)

        medianLoss = np.median(loss)
        lossInterval = np.quantile(loss, (0.16, 0.84))

        print (medianLoss, lossInterval)
    
if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--manifest', type = str,
                        default = "/home/dan/studies/NDLArForwardME/testManifest.yaml",
                        help = "network manifest yaml file")
    parser.add_argument('-v', '--verbose',
                        action = 'store_true',
                        help = "print extra debug messages")
    parser.add_argument('-t', '--trainMode',
                        action = 'store_true',
                        help = "run the evaluation loop in train mode instead of eval mode (useful for networks with dropout)")
    parser.add_argument('-n', '--nBatches',
                        default = 50,
                        type = int,
                        help = "Number of batches from the test dataset to evaluate on each checkpoint")
    parser.add_argument('-l', '--useLast',
                        action = 'store_true',
                        help = "optionally, use the last checkpoint in the epoch as a proxy")
    
    
    args = parser.parse_args()
    
    main(args)
