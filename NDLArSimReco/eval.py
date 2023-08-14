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

    entryLen = 4
    print (checkpointDict)
    if 'medianAccuracy' in checkpointDict[1]:
        entryLen += 3
    if 'lossList' in checkpointDict[1]:
        entryLen += len(checkpointDict[1]['lossList'])

    outArray = np.ndarray((entryLen, len(epoch)))

    if 'medianAccuracy' in checkpointDict[1]:
        medAccuracy = np.array([thisValue['medianAccuracy']
                                for thisValue in checkpointDict.values()])
        accInterval = np.array([thisValue['accInterval']
                                for thisValue in checkpointDict.values()])
        outArray[4,:] = medAccuracy
        outArray[5,:] = accInterval[:,0]
        outArray[6,:] = accInterval[:,1]
    if 'lossList' in checkpointDict[1]:
        lossArray = np.array([thisValue['lossList']
                              for thisValue in checkpointDict.values()])
        outArray[7:, :] = lossArray.T
    
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
    net = ConfigurableSparseNetwork(D=3, manifest = manifest).to(device)

    if args.trainMode:
        mode = 'train'
    elif args.dropoutMode:
        mode = 'MCdropout'
    else:
        mode = 'eval'

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

    evaluatedCheckpoints = {}

    for log_entry in net.log_manager.entries:
        if log_entry['n_iter'] == 0:
            print ("loading from log entry:", log_entry.outDir)
            net.load_checkpoint(log_entry['checkpointPath'])
            result = net.evalLoop(dl, 
                                  args.nBatches, 
                                  accuracy = args.accuracy,
                                  evalMode = mode)
            
            if args.accuracy:
                loss, accuracy = result
            else:
                loss = result

            medianLoss = np.median(loss)
            lossInterval = np.quantile(loss, (0.16, 0.84))

            evaluatedCheckpoints[log_entry['n_epoch']] = {'medianLoss': medianLoss,
                                                          'lossInterval': lossInterval,
                                                          # 'lossList': loss,
            }
            if args.accuracy:
                medianAcc = np.median(accuracy)
                accInterval = np.quantile(accuracy, (0.16, 0.84))

                evaluatedCheckpoints[log_entry['n_epoch']]['medianAccuracy'] = medianAcc
                evaluatedCheckpoints[log_entry['n_epoch']]['accInterval'] = accInterval
                print (evaluatedCheckpoints[log_entry['n_epoch']])#['accuracy'] = accuracy
                print (medianLoss, lossInterval, accuracy)
            else:
                print (medianLoss, lossInterval)

    save_record(manifest, evaluatedCheckpoints)
    
if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--manifest', type = str,
                        default = "/home/dan/studies/NDLArForwardME/testManifest.yaml",
                        help = "network manifest yaml file")
    parser.add_argument('-v', '--verbose',
                        action = 'store_true',
                        help = "print extra debug messages")
    parser.add_argument('-a', '--accuracy',
                        action = 'store_true',
                        help = "whether to calculate the accuracy alongside the batch loss")
    parser.add_argument('-t', '--trainMode',
                        action = 'store_true',
                        help = "run the evaluation loop in train mode instead of eval mode (useful for networks with dropout)")
    parser.add_argument('-d', '--dropoutMode',
                        action = 'store_true',
                        help = "run the evaluation loop in MC dropout mode instead of eval mode (all modules in eval mode, except dropout modules are in train mode)")
    parser.add_argument('-n', '--nBatches',
                        default = 50,
                        type = int,
                        help = "Number of batches from the test dataset to evaluate on each checkpoint")
    parser.add_argument('-l', '--useLast',
                        action = 'store_true',
                        help = "optionally, use the last checkpoint in the epoch as a proxy")
    
    
    args = parser.parse_args()
    
    main(args)
