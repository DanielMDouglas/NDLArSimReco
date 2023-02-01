import numpy as np
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

def plot_edep(tensor, label = None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    
    scatter = ax.scatter(*tensor.coordinates.T[1:,:],
                         c = tensor.features.detach().numpy()[:,0],
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

def criterionOld(output, truth):
    lossDomain = output.coordinates.float()
    truthFeat = truth.features_at_coordinates(lossDomain)
    predFeat = output.features_at_coordinates(lossDomain)
    return nn.MSELoss()(predFeat, truthFeat)

def criterion(output, truth):
    diff = (output - truth).features
    return nn.MSELoss()(diff, torch.zeros_like(diff))

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

    dl = DataLoader(infileList, batchSize = 1)
    dl.setFileLoadOrder()

    net = ConfigurableSparseNetwork(in_feat = 1,
                                    out_feat = 1,
                                    D = 3,
                                    manifest = manifest).to(device)
    net.load_checkpoint(args.checkpoint)

    dl.loadNextFile(0)

    theseHits, theseTracks = dl.load_event(3)
    
    hitList = [theseHits]
    trackList = [theseTracks]
    
    hitCoordTensors = []
    hitFeatureTensors = []

    trackCoordTensors = []
    trackFeatureTensors = []

    for hits, tracks in zip(hitList, trackList):

        trackX, trackZ, trackY, dE = tracks
        trackCoords = torch.FloatTensor(np.array([trackX, trackY, trackZ])).T.int()
        trackFeature = torch.FloatTensor(np.array([dE])).T

        trackdEthreshold = 0.25
        thresholdMask = (trackFeature > trackdEthreshold).flatten()

        trackCoords = trackCoords[thresholdMask]
        trackFeature = trackFeature[thresholdMask]

        trackCoordTensors.append(trackCoords)
        trackFeatureTensors.append(trackFeature)

        hitsX, hitsY, hitsZ, hitsQ = hits
        hitCoordsFilled = torch.FloatTensor(np.array([hitsX, hitsY, hitsZ])).T.int()
        hitFeatureFilled = torch.FloatTensor(np.array([hitsQ])).T

        # hitCoordsUnfilled = torch.stack([thisCoord
        #                                  for thisCoord in trackCoords])

        # hitCoordsFull = torch.concat((hitCoordsFilled, hitCoordsUnfilled))
        
        # nEmpty = hitCoordsUnfilled.shape[0] 
        # paddedFeats = torch.concat((hitFeatureFilled, torch.zeros((nEmpty, 1))))

        # hitCoords = hitCoordsFull
        # hitFeature = paddedFeats
        hitCoords = hitCoordsFilled
        hitFeature = hitFeatureFilled
        
        hitCoordTensors.append(hitCoords)
        hitFeatureTensors.append(hitFeature)

        
    hitCoords, hitFeature = ME.utils.sparse_collate(hitCoordTensors, 
                                                    hitFeatureTensors,
                                                    dtype = torch.int32)

    print (hitCoords.shape)
    
    trackCoords, trackFeature = ME.utils.sparse_collate(trackCoordTensors, 
                                                        trackFeatureTensors,
                                                        dtype = torch.int32)

    emptyFeats = [torch.zeros_like(thisTrackFeatureTensor)
                  for thisTrackFeatureTensor in trackFeatureTensors]
    
    trackCoordsEmpty, trackFeatureEmpty = ME.utils.sparse_collate(trackCoordTensors, 
                                                                  emptyFeats,
                                                                  dtype = torch.int32)
    
    larpix = ME.SparseTensor(features = hitFeature.to(device),
                             coordinates = hitCoords.to(device))
    edep = ME.SparseTensor(features = trackFeature.to(device),
                           coordinates = trackCoords.to(device))
    edepEmpty = ME.SparseTensor(features = trackFeatureEmpty.to(device),
                                coordinates = trackCoordsEmpty.to(device))

    larpix = larpix + edepEmpty

    # do some training loops on just this image -- see if I can learn a single thing
    optimizer = optim.SGD(net.parameters(), lr = 1.e-2, momentum = 0.9)

    prediction = net(larpix)
    plot_edep(prediction, 'prediction before training')

    for i in range(100):
        optimizer.zero_grad()
        
        prediction = net(larpix)

        loss = criterion(prediction, edep)
        loss.backward()
        optimizer.step()
        print (i, loss.item())
        with open("trainLog", 'a') as logFile:
            # with open("unionDomainLog", 'a') as logFile:
            logFile.write('{} \t {} \n'.format(i, loss.item()))

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
                        default = "./checkpoint_1_540.ckpt",
                        help = "network manifest yaml file")
    
    
    args = parser.parse_args()
    
    main(args)

