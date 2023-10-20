import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import random
import numpy as np
import h5py
import tqdm
import particle

from NDLArSimReco.network import ConfigurableSparseNetwork
from NDLArSimReco.dataLoader import *
from NDLArSimReco.utils import sparseTensor

import yaml
import os

def main(args):
    with open(args.manifest) as mf:
        manifest = yaml.load(mf, Loader = yaml.FullLoader)

    print ("initializing network...")
    net = ConfigurableSparseNetwork(D=3, manifest = manifest).to(device)

    if args.test:
        infilePath = manifest['testfilePath'] 
    else:
        infilePath = manifest['trainfilePath'] 
    if os.path.isdir(infilePath[0]):
        infileList = [os.path.join(infilePath[0], thisFile) 
                      for thisFile in os.listdir(infilePath[0])]
        print ("loading files from list", infileList)
    else:
        infileList = infilePath
        print ("loading files from list", infileList)

    print ("initializing data loader...")
    dl = dataLoaderFactory[manifest['dataLoader']]([infileList[0]],
                                                   batchSize = manifest['batchSize'],
                                                   sequentialLoad = True)
    net.log_manager.dataLoader = dl

    if args.checkpoint:
        try:
            print ("loading from checkpoint", args.checkpoint)
            net.load_checkpoint(args.checkpoint)                        
        except IOError:
            print ("could not load from checkpoint!")
    elif any(net.log_manager.entries):
        latestCheckpoint = net.log_manager.entries[-1]
        latestCheckpoint.load()
        print ("loading checkpont at epoch {}, iteration {}".format(net.n_epoch, net.n_iter))

    # net.eval()
    # net.train()
    net.MCdropout()

    dl.genFileLoadOrder()
    print ("fileLoadOrder", dl.fileLoadOrder)
    for fileIndex in tqdm.tqdm(dl.fileLoadOrder):
        try:
            dl.loadNextFile(fileIndex)
            print (dl.currentFileName)
        except OSError:
            print ("skipping bad file...")
            continue

        transform = sparseTensor.transformFactory[manifest['transform']](augment = False)
        
        dl.genSampleLoadOrder()
        # pbar = tqdm.tqdm(dl.sampleLoadOrder[:])
        pbar = tqdm.tqdm([0])
        for evIndex in pbar:
            hits, edep = dl.load_image(evIndex)
            evinfo = dl.evinfo_ev

            # print (hits.shape, edep.shape)

            # hitsST, edepST = transform(hits, edep)
            hitsST, edepST = transform([hits], [edep])

            # output = net.forward(hitsST)
            # loss = net.criterion(output, edepST)
 
            # pbarMessage = " ".join(["loss:",
            #                         str(round(loss.item(), 4))])
            # pbar.set_description(pbarMessage)
            # inference = net.criterion.feature_map(output)
                
if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--manifest', type = str,
                        required = True,
                        help = "network manifest yaml file")
    parser.add_argument('-c', '--checkpoint', type = str,
                        required = False,
                        help = "checkpoint file to start from")
    parser.add_argument('-t', '--test',
                        action = 'store_true',
                        help = "checkpoint file to start from")
    parser.add_argument('-o', '--outdir', type = str,
                        required = False,
                        help = "output")
    
    args = parser.parse_args()

    main(args)
