import torch
# torch.manual_seed(12)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import random
# random.seed(12)

from NDLArSimReco.network import ConfigurableSparseNetwork
# from NDLArSimReco.dataLoader import DataLoader
from NDLArSimReco.dataLoader import dataLoaderFactory
from NDLArSimReco.utils import sparseTensor

import yaml
import os
import tqdm

import matplotlib.pyplot as plt
import numpy as np

def main(args):
    with open(args.manifest) as mf:
        manifest = yaml.load(mf, Loader = yaml.FullLoader)

    print ("initializing network...")
    net = ConfigurableSparseNetwork(D=3, manifest = manifest).to(device)
    # net = ConfigurableSparseNetwork(in_feat=1, D=3, manifest = manifest).to(device)
    # net = ConfigurableSparseNetwork(in_feat=2, D=3, manifest = manifest).to(device)

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
    # dl = dataLoaderFactory[manifest['dataLoader']](infileList,
    #                                                batchSize = 10)
    net.log_manager.dataLoader = dl
    
    print ("loading from checkpoint", args.checkpoint)
    net.load_checkpoint(args.checkpoint)

    net.eval()

    LABELS = [11,22,13,211,2212]

    transform = sparseTensor.transformFactory[manifest['transform']]
    pbar = tqdm.tqdm(enumerate(dl.load(transform = transform)),
                     total = dl.batchesPerEpoch)
    for i, (inpt, truth) in pbar:
        output = net.forward(inpt)
        inference = torch.argmax(output.features, axis = 1)
        print (output.features)
        print (inference)
        print (truth)
        
        print (net.criterion(output, truth))
        
        fig = plt.figure()
        plt.hist2d(inference.numpy(), truth.numpy(),
                   bins = (np.linspace(-0.5, 4.5, 6),
                           np.linspace(-0.5, 4.5, 6),
                           ))
        plt.savefig('confusion.png')
        
        
if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--manifest', type = str,
                        default = "/sdf/home/d/dougl215/studies/NDLArSimReco/NDLArSimReco/manifests/testManifest.yaml",
                        help = "network manifest yaml file")
    parser.add_argument('-c', '--checkpoint', type = str,
                        default = "",
                        help = "checkpoint file to start from",
                        required = True)
    
    args = parser.parse_args()

    main(args)
