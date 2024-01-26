import torch
# torch.manual_seed(12)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import random
# random.seed(12)

import numpy as np
# np.random.seed(12)
import scipy.stats as st
        
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

    print ("initializing network...")
    net = ConfigurableSparseNetwork(D=3, manifest = manifest).to(device)

    infilePath = manifest['testfilePath'] 
    if os.path.isdir(infilePath[0]):
        infileList = sum(([os.path.join(thisPath, thisFile) 
                           for thisFile in os.listdir(thisPath)]
                           for thisPath in infilePath),
                          start = [])
        print ("loading files from list", infileList)
    else:
        infileList = infilePath
        print ("loading files from list", infileList)

    print ("initializing data loader...")
    dl = dataLoaderFactory[manifest['dataLoader']](infileList,
                                                   # batchSize = manifest['batchSize'],
                                                   batchSize = 1,
                                                   sequentialLoad = True)

    net.load_checkpoint(args.checkpoint)

    # net.MCdropout()
    net.eval()
    nBatches = args.nBatches

    transform = sparseTensor.transformFactory[manifest['transform']](augment = False)
    pbar = tqdm.tqdm(enumerate(dl.load(transform = transform)),
                     total = nBatches)

    trueE = []
    infE = []
    depE = []

    for i, (inpt, truth) in pbar:
        if i >= nBatches:
            break # we're done here

        output = net.forward(inpt)
        loss = net.criterion(output, truth)
        
        pbar.set_description("loss: "+str(round(loss.item(), 4)))
        # print("result loss", round(loss.item(), 4))        
        # loss = net.criterion(torch.ones_like(output), torch.ones_like(truth))
        # print("fake loss", round(loss.item(), 4))

        # print (output.features)
        inferredE = output.features[:,0]

        # print (truth)
        # print (transform.totE)

        # print (inferredE)
        truthArr = truth.cpu().numpy().flatten()
        infArr = inferredE.detach().cpu().numpy().flatten()
        # print (truthArr)
        # print (infArr)
        trueE.append(truthArr)
        infE.append(infArr)
        
        # print (np.abs(truthArr - infArr)/truthArr)
        # print (np.max(np.abs(truthArr - infArr)/truthArr))
        # maxMask = np.max(np.abs(truthArr - infArr)/truthArr) == np.abs(truthArr - infArr)/truthArr
        # print (truthArr[maxMask], infArr[maxMask])
        # if np.abs(truthArr[maxMask][0] - infArr[maxMask][0])/truthArr[maxMask][0] > 5:
        #     print ("thing")
        #     torch.save(inpt.coordinates, 'funky_coords')
        #     torch.save(inpt.features, 'funky_feats')
        
        # print (truth, inferredE)

    print (trueE)
    print (infE)
    
    trueE = np.concatenate(trueE)
    infE = np.concatenate(infE)

    print (trueE)
    print (infE)

    print (np.max(np.abs(trueE - infE)/trueE))

    Ebins = np.linspace(0, 1100, 23)

    fig = plt.figure()
    plt.hist2d(trueE,
               infE,
               bins = (Ebins, Ebins))
    plt.xlabel(r'True Primary Energy [MeV]')
    plt.ylabel(r'Inferred Primary Energy [MeV]')

    plt.savefig(args.outPrefix+'_true_inf_hist2d.png')

    fracErrBins = np.linspace(0, 2, 21)
    fig = plt.figure()
    plt.hist2d(trueE,
               np.abs(infE-trueE)/trueE,
               bins = (Ebins, fracErrBins))
    plt.xlabel(r'True Primary Energy [MeV]')
    plt.ylabel(r'Absolute Relative Error')

    bs_mean = st.binned_statistic(trueE, np.abs(infE-trueE)/trueE, bins = Ebins, statistic = 'mean')
    binCenters = 0.5*(Ebins[:-1] + Ebins[1:])
    plt.scatter(binCenters, bs_mean[0],
                marker = '+',
                color = 'red')

    plt.savefig(args.outPrefix+'_true_err_hist2d.png')

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
    parser.add_argument('-c', '--checkpoint',
                        help = "load this checkpoint")
    parser.add_argument('-n', '--nBatches',
                        default = 50,
                        type = int,
                        help = "Number of batches from the test dataset to evaluate on each checkpoint")
    parser.add_argument('-o', '--outPrefix',
                        default = 'test_',
                        type = str,
                        help = "prefix string for output plots")
    
    args = parser.parse_args()
    
    main(args)
