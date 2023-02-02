import torch
# torch.manual_seed(12)

import random
# random.seed(12)

from NDLArSimReco.network import ConfigurableSparseNetwork
from NDLArSimReco.dataLoader import DataLoader

import yaml
import os
            
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(args.manifest) as mf:
        manifest = yaml.load(mf, Loader = yaml.FullLoader)

    print ("initializing network...")
    net = ConfigurableSparseNetwork(in_feat=1, out_feat=1, D=3, manifest = manifest).to(device)

    if args.checkpoint:
        try:
            net.load_checkpoint(args.checkpoint)
        except IOError:
            print ("could not load from checkpoint!")
    
    infilePath = manifest['trainfilePath'] 
    if os.path.isdir(infilePath[0]):
        infileList = [os.path.join(infilePath[0], thisFile) 
                      for thisFile in os.listdir(infilePath[0])]
        print ("loading files from list", infileList)
    else:
        infileList = infilePath
        print ("loading files from list", infileList)
    print ("initializing data loader...")
    dl = DataLoader(infileList, batchSize = manifest['batchSize'])
    
    if args.force:
        # remove previous checkpoints
        for oldCheckpoint in os.listdir(os.path.join(manifest['outdir'],
                                                     'checkpoints')):
            print ('removing ', oldCheckpoint)
            os.remove(os.path.join(manifest['outdir'],
                                   'checkpoints',
                                   oldCheckpoint))
        net.manifest['checkpoints'] = []
        reportFile = os.path.join(manifest['outdir'],
                                  'train_report.dat')
        if os.path.exists(reportFile):
            os.remove(reportFile)
        
    print ("training...")
    net.train(dl)

    checkpointFile = os.path.join(net.outDir,
                                  'checkpoint_final_{}_{}.ckpt'.format(manifest['nEpochs'], 0))
    net.make_checkpoint(checkpointFile)

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
