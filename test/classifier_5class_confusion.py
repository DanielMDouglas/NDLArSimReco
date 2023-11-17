import torch
# torch.manual_seed(12)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import random
# random.seed(12)

import numpy as np
# np.random.seed(12)
import scipy.stats as st
from sklearn.metrics import *
        
from NDLArSimReco.network import ConfigurableSparseNetwork
from NDLArSimReco.dataLoader import dataLoaderFactory
from NDLArSimReco.utils import sparseTensor

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import yaml
import os
import tqdm

def main(args):
    with open(args.manifest) as mf:
        manifest = yaml.load(mf, Loader = yaml.FullLoader)
    with open(args.manifest2) as mf:
        manifest2 = yaml.load(mf, Loader = yaml.FullLoader)

    print ("initializing network...")
    net = ConfigurableSparseNetwork(D=3, manifest = manifest).to(device)

    print ("initializing network...")
    net2 = ConfigurableSparseNetwork(D=3, manifest = manifest2).to(device)

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
                                                   batchSize = manifest['batchSize'],
                                                   sequentialLoad = True)

    net.load_checkpoint(args.checkpoint)
    net2.load_checkpoint(args.checkpoint2)

    net.MCdropout()
    net2.MCdropout()
    # net.eval()
    # net2.eval()
    nBatches = args.nBatches

    transform = sparseTensor.transformFactory[manifest['transform']](augment = False)
    pbar = tqdm.tqdm(enumerate(dl.load(transform = transform)),
                     total = nBatches)

    truthList = []
    
    predList = []

    predList2 = []

    for i, (inpt, truth) in pbar:
        if i >= nBatches:
            break # we're done here

        output = net.forward(inpt)
        output2 = net2.forward(inpt)
        loss = net.criterion(output, truth)

        truthFeat = truth
        # mask = truthFeat != -9999
        truthFeat = truthFeat[:]
        # prediction = torch.sigmoid(output.features[:,0]) > 0.5
        # prediction2 = torch.sigmoid(output2.features[:,0]) > 0.5

        prediction = torch.argmax(output.features, dim = -1)
        prediction2 = torch.argmax(output2.features, dim = -1)
        # prediction = torch.argmax(output.features[:,::2], dim = -1)
        # prediction2 = torch.argmax(output2.features[:,::2], dim = -1)

        print (sum(prediction == truthFeat)/len(prediction))
        print (sum(prediction2 == truthFeat)/len(prediction2))
        
        print (truthFeat)
        print (prediction)
        truthList.append(truthFeat)
        
        predList.append(prediction)
        predList2.append(prediction2)

    truthList = np.concatenate(truthList)
        
    predList = np.concatenate(predList)
    predList2 = np.concatenate(predList2)

    cm = confusion_matrix(truthList, predList)
    # cm_display = ConfusionMatrixDisplay(cm,
    #                                     display_labels=['Track', 'Shower'],
    #                                     ).plot(cmap = 'Blues',
    #                                             normalize = True)
    cm_display = ConfusionMatrixDisplay.from_predictions(truthList, predList,
                                                         # display_labels=['Track', 'Shower'],
                                                         cmap = 'Blues',
                                                         normalize = 'true')
    plt.savefig('confusion_inhomog.png')

    cm2 = confusion_matrix(truthList, predList2)
    # cm_display2 = ConfusionMatrixDisplay(cm2,
    #                                      display_labels=['Track', 'Shower'],
    #                                      ).plot(cmap = 'Blues',
    #                                             normalize = True)
    cm_display2 = ConfusionMatrixDisplay.from_predictions(truthList, predList2,
                                                          # display_labels=['Track', 'Shower'],
                                                          cmap = 'Blues',
                                                          normalize = 'true')
    plt.savefig('confusion_homog.png')

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--manifest', type = str,
                        default = "/home/dan/studies/NDLArForwardME/testManifest.yaml",
                        help = "network manifest yaml file")
    parser.add_argument('-m2', '--manifest2', type = str,
                        default = "/home/dan/studies/NDLArForwardME/testManifest.yaml",
                        help = "network manifest yaml file")
    parser.add_argument('-v', '--verbose',
                        action = 'store_true',
                        help = "print extra debug messages")
    parser.add_argument('-a', '--accuracy',
                        action = 'store_true',
                        help = "whether to calculate the accuracy alongside the batch loss")
    parser.add_argument('-c', '--checkpoint',
                        help = "load this checkpoint")
    parser.add_argument('-c2', '--checkpoint2',
                        help = "load this checkpoint")
    parser.add_argument('-n', '--nBatches',
                        default = 50,
                        type = int,
                        help = "Number of batches from the test dataset to evaluate on each checkpoint")
    
    args = parser.parse_args()
    
    main(args)
