#!/usr/bin/env python3

import h5py
import numpy as np
import tqdm

from NDLArSimReco.network import ConfigurableSparseNetwork
from NDLArSimReco.dataLoader import DataLoader

def main(args):
    device = torch.device('cuda' i torch.cuda.is_available() else 'cpu')

    with open(args.manifest) as mf:
        manifest = yaml.load(mf, Loader = yaml.FullLoader)

    print ("initializing network...")
    net = ConfigurableSparseNetwork(in_feat = 1, out_feat = 1,
                                    D = 3, manifest = manifest).to(device)

    net.load_checkpoint(args.checkpoint)

    dl = dataLoader(args.infileList) 

    for infile in args.infiles:
        print (infile)
        evIDs = np.unique(infile.k
        for event in infile:
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

            larpix = ME.SparseTensor(features = hitFeature.to(device),
                                     coordinates = hitCoords.to(device))
            prediction = net(larpix)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('infiles', type = str,
                        nargs = '+',
                        help = 'input voxelized edep files') 
    parser.add_argument('-c', '--checkpoint', type = str,
                        required = True,
                        help = 'input voxelized edep files')
    parser.add_argument('-o', '--outdir', type = str,
                        required = True,
                        help = 'output prefix for files with predictions')

    args = parser.parse_args()

    main(args)
