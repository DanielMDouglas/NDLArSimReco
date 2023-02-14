import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

import random

from NDLArSimReco.network import ConfigurableSparseNetwork
from NDLArSimReco.dataLoader import DataLoader

import yaml
import os

import MinkowskiEngine as ME
ME.set_sparse_tensor_operation_mode(ME.SparseTensorOperationMode.SHARE_COORDINATE_MANAGER)

def plot_edep(tensor, label = None, featInd = 0):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    
    scatter = ax.scatter(*tensor.coordinates.T[1:,:],
                         c = tensor.features.detach().numpy()[:,featInd],
                         label = label,
                         )
    ax.legend(frameon = False)
    ax.set_xlabel(r'x [mm]')
    ax.set_ylabel(r'y [mm]')
    ax.set_zlabel(r'z [mm]')

    plt.colorbar(scatter)

def plot_edep_multi(tensorList, labelList, colors):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')

    for i, (thisTensor, thisLabel, thisColor) in enumerate(zip(tensorList, labelList, colors)):
        scatter = ax.scatter(*thisTensor.coordinates.T[1:,:],
                             label = thisLabel,
                             zorder = 4 - i,
                             alpha = 0.2,
                             color = thisColor,
                             )
    ax.legend(frameon = False)
    ax.set_xlabel(r'x [mm]')
    ax.set_ylabel(r'y [mm]')
    ax.set_zlabel(r'z [mm]')

from NDLArSimReco.loss import MSE 
# from NDLArSimReco.loss import NLL_homog as criterion 
from NDLArSimReco.loss import NLL as criterion 

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(args.manifest) as mf:
        manifest = yaml.load(mf, Loader = yaml.FullLoader)

    infilePath = manifest['trainfilePath']
    if os.path.isdir(infilePath[0]):
        infileList = [os.path.join(infilePath[0], thisFile)
                      for thisFile in os.listdir(infilePath[0])]
    else:
        infileList = infilePath

    dl = DataLoader(infileList, batchSize = 10)
    dl.setFileLoadOrder()

    net = ConfigurableSparseNetwork(in_feat = 1,
                                    D = 3,
                                    manifest = manifest).to(device)
    if args.checkpoint:
        try:
            net.load_checkpoint(args.checkpoint)
        except IOError:
            print ("could not load from checkpoint!")

    dl.loadNextFile(0)

    # larpix, edep = dl.load_event(5)
    # larpix, edep = dl.load_event(3)
    larpix, edep = next(dl.load())
    
    optimizer = optim.SGD(net.parameters(), lr = 1.e-2, momentum = 0.9)

    # prediction = net(larpix)
    # plot_edep(prediction, 'prediction before training')

    # loss = criterion(edep, edep)
    # print ("self loss:", loss.item())

    for i in range(100):
        # larpix, edep = next(dl.load())

        optimizer.zero_grad()
        
        prediction = net(larpix)

        loss = criterion(prediction, edep)
        MSEloss = MSE(prediction, edep)
        loss.backward()
        optimizer.step()
        print ("iter:", i,
               "NLL loss:", loss.item(),
               "MSE loss:", MSEloss.item(),
               end = '\r')
        with open("trainLog", 'a') as logFile:
            # with open("unionDomainLog", 'a') as logFile:
            logFile.write('{} \t {} \n'.format(i, loss.item()))

    print ("final loss:", loss.item())

    net.make_checkpoint('single_track.ckpt')
    plot_edep(prediction, 'prediction')
    plot_edep(edep, 'truth')
    plot_edep(larpix, 'larpix')
    plot_edep(prediction - edep, 'diff')
    plot_edep_multi([larpix, edep],
                    ['larnd-sim', 'edep-sim'],
                    ['black', 'red'])
    
    plt.show()

if __name__ == '__main__':
    
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--manifest', type = str,
                        default = "/home/dan/studies/NDLArSimReco/NDLArSimReco/manifests/localTestManifest.yaml",
                        help = "network manifest yaml file")
    parser.add_argument('-c', '--checkpoint', type = str,
                        default = "",
                        help = "checkpoint file to start from")
    
    args = parser.parse_args()
    
    main(args)

