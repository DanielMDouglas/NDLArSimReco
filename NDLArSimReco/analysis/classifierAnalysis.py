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
import particle

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
    #                                                batchSize = 1)
    net.log_manager.dataLoader = dl
    
    print ("loading from checkpoint", args.checkpoint)
    net.load_checkpoint(args.checkpoint)

    net.eval()

    LABELS = [11,22,13,211,2212]
    LaTeXlabels = ['$'+particle.Particle.from_pdgid(i).latex_name+'$' for i in LABELS]

    # inferences = []
    # true_labels = []
    inferences = np.empty((0,))
    true_labels = np.empty((0,))
    
    transform = sparseTensor.transformFactory[manifest['transform']]
    pbar = tqdm.tqdm(enumerate(dl.load(transform = transform)),
                     total = dl.batchesPerEpoch)
    for i, (inpt, truth) in pbar:
        # if i > 1000:
        #     break
        
        output = net.forward(inpt)
        thisInference = torch.argmax(output.features, axis = 1)

        print ("loss", net.criterion(output, truth))
        
        # inferences.append(inference.item())
        # true_labels.append(truth.item())
        inferences = np.concatenate((inferences, thisInference))
        true_labels = np.concatenate((true_labels, truth))

    H, xedges, yedges = np.histogram2d(inferences, true_labels,
                                       bins = (np.linspace(-0.5, 4.5, 6),
                                               np.linspace(-0.5, 4.5, 6),
                                               ))
    H = H.T

    fig = plt.figure()
    plt.imshow(H, origin = 'upper',
               extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]],
               cmap = 'Blues')
    fig.gca().tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    fig.gca().xaxis.set_label_position('top')

    plt.xticks(range(5), LaTeXlabels)
    plt.yticks(range(5), LaTeXlabels[::-1])

    plt.xlabel(r'Inferred Label')
    plt.ylabel(r'True Label')
    
    print (H)
    for i, row in enumerate(H):
        for j, value in enumerate(row):
            percent = int(100*value/sum(row))
            text = str(percent)+'%\n('+str(int(value))+')'
            if value > 0.6*np.max(H):
                color = 'white'
            else:
                color = '#08306B'
            plt.text(i, len(LABELS)-j-1,
                     text,
                     color = color,
                     ha = 'center',
                     va = 'center')
    
    # plt.colorbar()
    plt.savefig(args.outfile)
        
        
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
    parser.add_argument('-o', '--outfile', type = str,
                        default = "",
                        help = "output image path",
                        required = True)
    
    args = parser.parse_args()

    main(args)
