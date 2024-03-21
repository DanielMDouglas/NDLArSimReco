import torch
# torch.manual_seed(12)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import random
# random.seed(12)

from NDLArSimReco.network import ConfigurableSparseNetwork
# from NDLArSimReco.dataLoader import DataLoader
from NDLArSimReco.dataLoader import dataLoaderFactory

import yaml
import os

def main(args):
    with open(args.manifest) as mf:
        manifest = yaml.load(mf, Loader = yaml.FullLoader)

    print ("initializing network...")
    net = ConfigurableSparseNetwork(D=3, manifest = manifest).to(device)

    infilePath = manifest['trainfilePath'] 
    if all(os.path.isdir(thisPath) for thisPath in infilePath):
        infileList = sum(([os.path.join(thisPath, thisFile) 
                           for thisFile in os.listdir(thisPath)]
                           for thisPath in infilePath),
                         start = [])
        print ("loading files from directory", infilePath)
    else:
        infileList = infilePath
        print ("loading files from list", infileList)

    print ("initializing data loader...")
    # dl = DataLoader(infileList, batchSize = manifest['batchSize'])
    dl = dataLoaderFactory[manifest['dataLoader']](infileList,
                                                   batchSize = manifest['batchSize'])
    net.log_manager.dataLoader = dl
    
    if args.force:
        # remove previous checkpoints
        net.log_manager.clear()
    elif args.checkpoint:
        try:
            print ("loading from checkpoint", args.checkpoint)
            net.log_manager.revert_state(args.checkpoint)            
            # for thisEntry in net.log_manager.entries:
            #     if os.path.abspath(args.checkpoint) == os.path.abspath(thisEntry.outDir):
            #         print("found matching entry:", thisEntry.outDir)
            #         thisEntry.load()
                
        except IOError:
            print ("could not load from checkpoint!")
    elif any(net.log_manager.entries):
        latestCheckpoint = net.log_manager.entries[-1]
        latestCheckpoint.load()

        print ("resuming training at epoch {}, iteration {}".format(net.n_epoch, net.n_iter))
            
    print ("training...")
    net.trainLoop(dl, verbose = args.verbose)

    checkpointFile = os.path.join(net.outDir,
                                  'checkpoint_final_{}_{}.ckpt'.format(manifest['nEpochs'], 0))
    net.make_checkpoint(checkpointFile)
    net.log_manager.save_report()

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--manifest', type = str,
                        default = "/sdf/home/d/dougl215/studies/NDLArSimReco/NDLArSimReco/manifests/testManifest.yaml",
                        help = "network manifest yaml file")
    parser.add_argument('-f', '--force',
                        action = 'store_true',
                        help = "forcibly train the network from scratch")
    parser.add_argument('-v', '--verbose',
                        action = 'store_true',
                        help = "print extra debug messages")
    parser.add_argument('-c', '--checkpoint', type = str,
                        default = "",
                        help = "checkpoint file to start from")
    
    args = parser.parse_args()

    main(args)
