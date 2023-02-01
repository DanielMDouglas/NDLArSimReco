import numpy as np

import matplotlib.pyplot as plt

import tqdm

import torch
# torch.manual_seed(12)

import random
# random.seed(12)

from NDLArSimReco.network import ConfigurableSparseNetwork
from NDLArSimReco.dataLoader import DataLoader

import yaml
import os

import MinkowskiEngine as ME

# from SLACplots import colors

def criterion(output, truth):
    gaussWidth = 10
    jointLogLikelihood = 0
    totalEMD = 0
    totalEtrue = []
    totalEpred = []

    for batchNo in torch.unique(output.coordinates[:,0]):
        batchPredCoords = output.coordinates[output.coordinates[:,0] == batchNo][:,1:].float()
        batchTrueCoords = truth.coordinates[truth.coordinates[:,0] == batchNo][:,1:].float()

        nTrue = batchTrueCoords.shape[0]
        nPred = batchPredCoords.shape[0]

        batchPredE = output.features[output.coordinates[:,0] == batchNo][:,0]
        # batchPredProb = output.features[output.coordinates[:,0] == batchNo][:,1]

        batchTrueE = truth.features[truth.coordinates[:,0] == batchNo]
                
        normedBatchPredE = torch.abs(batchPredE/torch.sum(batchPredE))
        # normedBatchPredE = batchPredE
        normedBatchTrueE = batchTrueE/torch.sum(batchTrueE)

        mags = torch.prod(torch.stack((normedBatchPredE.repeat((nTrue, 1)),
                                 torch.swapaxes(normedBatchTrueE.flatten().repeat((nPred, 1)), 0, 1))),
                       0)
        distances = torch.linalg.norm(torch.sub(batchPredCoords.repeat((nTrue, 1, 1)),
                                          torch.swapaxes(batchTrueCoords.repeat((nPred, 1, 1)), 0, 1)),
                                   dim = 2)
        probs = torch.sum(mags*torch.exp(-torch.pow(distances/(2*gaussWidth), 2)), 0)/(gaussWidth*np.sqrt(2*torch.pi))
        jointLogLikelihood += torch.sum(torch.log(probs))/nPred
        
    nLogL = -jointLogLikelihood

    return nLogL

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(args.manifest) as mf:
        manifest = yaml.load(mf, Loader = yaml.FullLoader)

    # print ("initializing network...")
    # net = ConfigurableSparseNetwork(in_feat=1, out_feat=1, D=3, manifest = manifest).to(device)

    infilePath = manifest['trainfilePath'] 
    if os.path.isdir(infilePath[0]):
        infileList = [os.path.join(infilePath[0], thisFile) 
                      for thisFile in os.listdir(infilePath[0])]
        print ("loading files from list", infileList)
    else:
        infileList = infilePath
        print ("loading files from list", infileList)
    print ("initializing data loader...")
    dl = DataLoader(infileList)
    dl.setFileLoadOrder()

    net = ConfigurableSparseNetwork(in_feat=1, out_feat=1, D=3, manifest = manifest).to(device)
    net.load_checkpoint(args.checkpoint)
    
    # for i, (hitList, trackList) in tqdm.tqdm(enumerate(dl.load()), total = dl.batchesPerEpoch):
    dl.loadNextFile(0)
    dLLH = []
    for eventID in range(5):
        theseHits, theseTracks = dl.load_event(eventID)

        hitList = [theseHits]
        trackList = [theseTracks]
    
        hitCoordTensors = []
        hitFeatureTensors = []

        for hits in hitList:
            hitsX, hitsY, hitsZ, hitsQ = hits
            hitCoords = torch.FloatTensor([hitsX, hitsY, hitsZ]).T
            hitFeature = torch.FloatTensor([hitsQ]).T
            
            hitCoordTensors.append(hitCoords)
            hitFeatureTensors.append(hitFeature)
            
        hitCoords, hitFeature = ME.utils.sparse_collate(hitCoordTensors, 
                                                        hitFeatureTensors,
                                                        dtype = torch.int32)
                
        trackCoordTensors = []
        trackFeatureTensors = []
        for tracks in trackList:
            trackX, trackZ, trackY, dE = tracks
            trackCoords = torch.FloatTensor([trackX, trackY, trackZ]).T
            trackFeature = torch.FloatTensor([dE]).T
                
            trackCoordTensors.append(trackCoords)
            trackFeatureTensors.append(trackFeature)
            
        trackCoords, trackFeature = ME.utils.sparse_collate(trackCoordTensors, 
                                                            trackFeatureTensors,
                                                            dtype = torch.int32)

    
        larpix = ME.SparseTensor(features = hitFeature.to(device),
                                 coordinates = hitCoords.to(device))
        edep = ME.SparseTensor(features = trackFeature.to(device),
                               coordinates = trackCoords.to(device))

        nomLoss = criterion(edep, edep)
        print ("self loss", nomLoss)

        # shiftSpace = np.linspace(-10, 10, 21)
        shiftSpace = np.linspace(-10, 10, 3)

        thisdLLR = []
        for rShift in shiftSpace:
        
            # shiftVec = torch.Tensor([0,
            #                          random.random(),
            #                          random.random(),
            #                          random.random()])
            shiftVec = torch.Tensor([0, rShift, 0, 0])
            # shiftVec *= rShift/torch.linalg.norm(shiftVec)
            shiftedTrackCoords = trackCoords + shiftVec
            edepShifted = ME.SparseTensor(features = trackFeature.to(device),
                                          coordinates = shiftedTrackCoords.to(device))
        
            # print ("unshiftedCoords:", trackCoords)
            # print ("shiftedCoords:", shiftedTrackCoords)

            shiftedLoss = criterion(edep, edepShifted)
            print ("shift", shiftVec)
            print ("shift norm", torch.linalg.norm(shiftVec))
            print ("shift loss", shiftedLoss)
            print ("dLLH", shiftedLoss - nomLoss)

            thisdLLR.append(shiftedLoss - nomLoss)
            
            fig = plt.figure()
            ax = fig.add_subplot(111, projection = '3d')
            ax.scatter(*trackCoords.T[1:,:],
                       # color = colors.SLACred,
                       label = 'true dE',
                       )
            # ax.scatter(*shiftedTrackCoords.T[1:,:],
            #            # color = colors.SLACblue,
            #            label = 'shifted',
            #            )
            prediction = net(larpix)
            ax.scatter(*prediction.coordinates.T[1:,:],
                       # color = colors.SLACblue,
                       label = 'prediction',
                       )
            plt.legend(frameon = False)
            ax.set_xlabel(r'x [mm]')
            ax.set_ylabel(r'y [mm]')
            ax.set_zlabel(r'z [mm]')

            ax.set_xlim(1370, 1440)
            ax.set_ylim(870, 950)
            ax.set_zlim(-65, -35)

            fig = plt.figure()
            ax = fig.add_subplot(111, projection = '3d')
            ax.scatter(*trackCoords.T[1:,:],
                       # color = colors.SLACred,
                       label = 'true dE',
                       )
            plt.legend(frameon = False)
            ax.set_xlabel(r'x [mm]')
            ax.set_ylabel(r'y [mm]')
            ax.set_zlabel(r'z [mm]')

            ax.set_xlim(1370, 1440)
            ax.set_ylim(870, 950)
            ax.set_zlim(-65, -35)

            fig = plt.figure()
            ax = fig.gca()
            print ('trackFeature', trackFeature.shape)
            ax.hist(trackFeature.flatten())

            ax.set_xlabel(r'dE')
            
            fig = plt.figure()
            ax = fig.add_subplot(111, projection = '3d')
            threshold = 0.25
            thresholdMask = (trackFeature > threshold).flatten()
            aboveThresholdTrackCoords = trackCoords[thresholdMask]
            belowThresholdTrackCoords = trackCoords[~thresholdMask]
            ax.scatter(*aboveThresholdTrackCoords.T[1:,:],
                       color = 'green',
                       label = 'true dE',
                       )
            ax.scatter(*belowThresholdTrackCoords.T[1:,:],
                       color = 'red',
                       label = 'true dE',
                       )
            plt.legend(frameon = False)
            ax.set_xlabel(r'x [mm]')
            ax.set_ylabel(r'y [mm]')
            ax.set_zlabel(r'z [mm]')

            ax.set_xlim(1370, 1440)
            ax.set_ylim(870, 950)
            ax.set_zlim(-65, -35)

            fig = plt.figure()
            ax = fig.add_subplot(111, projection = '3d')
            prediction = net(larpix)
            ax.scatter(*prediction.coordinates.T[1:,:],
                       # color = colors.SLACblue,
                       label = 'prediction',
                       )
            plt.legend(frameon = False)
            ax.set_xlabel(r'x [mm]')
            ax.set_ylabel(r'y [mm]')
            ax.set_zlabel(r'z [mm]')

            ax.set_xlim(1370, 1440)
            ax.set_ylim(870, 950)
            ax.set_zlim(-65, -35)

            plt.show()

        # scaleSpace = np.linspace(0.5, 1.5, 21)

        # thisdLLR = []
        # for scale in scaleSpace:
        #     trackFeatureScaled = trackFeature*scale

        #     edepScaled = ME.SparseTensor(features = trackFeatureScaled.to(device),
        #                                  coordinates = trackCoords.to(device))

        #     scaledLoss = criterion(edepScaled, edep)
        #     print ("scaled loss", scaledLoss)
        #     print ("dLLH", scaledLoss - nomLoss)

        #     thisdLLR.append(scaledLoss - nomLoss)
            
        dLLH.append(thisdLLR)

    dLLH = np.array(dLLH)
    print (dLLH.shape)
    med = np.quantile(dLLH, 0.5, axis = 0)

    errLo = med - np.quantile(dLLH, 0.16, axis = 0)
    errHi = np.quantile(dLLH, 0.84, axis = 0) - med
    
    # plt.scatter(shiftSpace, dLLH)
    plt.errorbar(shiftSpace, med, yerr = [errLo, errHi], fmt = 'o')
    # plt.errorbar(scaleSpace, med, yerr = [errLo, errHi], fmt = 'o')
    
    plt.xlabel(r'$E/E_0$ [scale factor]')
    plt.ylabel(r'$\Delta LLH$')
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
    parser.add_argument('-v', '--verbose',
                        action = 'store_true',
                        help = "print extra debug messages")
    
    
    args = parser.parse_args()
    
    main(args)
