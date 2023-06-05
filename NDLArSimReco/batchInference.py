import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import random
import numpy as np
import h5py
import tqdm

from NDLArSimReco.network import ConfigurableSparseNetwork
from NDLArSimReco.dataLoader import DataLoader
from NDLArSimReco.utils import sparseTensor

import yaml
import os

output_dtypes = {"hits": np.dtype([("eventID", "u4"),
                                   ("x", "f4"),
                                   ("y", "f4"),
                                   ("z", "f4"),
                                   ("q", "f4")],
                                  align = True),
                 "edep": np.dtype([("eventID", "u4"),
                                   ("x", "f4"),
                                   ("y", "f4"),
                                   ("z", "f4"),
                                   ("dE", "f4")],
                                  align = True),
                 "inference": np.dtype([("eventID", "u4"),
                                        ("x", "f4"),
                                        ("y", "f4"),
                                        ("z", "f4"),
                                        ("dE", "f4"),
                                        ("dE_err", "f4")],
                                       align = True),
                 "evinfo": np.dtype([("eventID", "u4"),
                                     ("primaryPID", "i4")],
                                    align = True),
                 }

def write_to_output(outfile, evHits, evEdep, evInf, evEv):
    nHits_prev = len(outfile['hits'])
    nHits_ev = len(evHits)
    outfile['hits'].resize((nHits_prev + nHits_ev,))
    outfile['hits'][nHits_prev:] = evHits
    
    nEdep_prev = len(outfile['edep'])
    nEdep_ev = len(evEdep)
    outfile['edep'].resize((nEdep_prev + nEdep_ev,))
    outfile['edep'][nEdep_prev:] = evEdep

    nInf_prev = len(outfile['inference'])
    nInf_ev = len(evInf)
    outfile['inference'].resize((nInf_prev + nInf_ev,))
    outfile['inference'][nInf_prev:] = evInf

    nEv_prev = len(outfile['evinfo'])
    nEv_ev = len(evEv)
    outfile['evinfo'].resize((nEv_prev + nEv_ev,))
    outfile['evinfo'][nEv_prev:] = evEv

def main(args):
    with open(args.manifest) as mf:
        manifest = yaml.load(mf, Loader = yaml.FullLoader)

    print ("initializing network...")
    net = ConfigurableSparseNetwork(in_feat=1, D=3, manifest = manifest).to(device)

    infilePath = manifest['trainfilePath'] 
    if os.path.isdir(infilePath[0]):
        infileList = [os.path.join(infilePath[0], thisFile) 
                      for thisFile in os.listdir(infilePath[0])]
        print ("loading files from list", infileList)
    else:
        infileList = infilePath
        print ("loading files from list", infileList)

    print ("initializing data loader...")
    dl = DataLoader(infileList,
                    batchSize = manifest['batchSize'],
                    sequentialLoad = True)
    net.log_manager.dataLoader = dl

    if args.checkpoint:
        try:
            print ("loading from checkpoint", args.checkpoint)
            net.log_manager.revert_state(args.checkpoint)            
            
        except IOError:
            print ("could not load from checkpoint!")
    elif any(net.log_manager.entries):
        latestCheckpoint = net.log_manager.entries[-1]
        latestCheckpoint.load()

        print ("loading checkpont at epoch {}, iteration {}".format(net.n_epoch, net.n_iter))

    net.eval()

    for fileIndex in tqdm.tqdm(dl.fileLoadOrder):
        dl.loadNextFile(fileIndex)
        print (dl.currentFileName)
        outputFileName = os.path.join(args.outdir,
                                      os.path.basename(dl.currentFileName))
        print ("trying to write to output file,", outputFileName)
        
        outfile = h5py.File(outputFileName, 'w')

        for key, value in output_dtypes.items():
            outfile.create_dataset(key,
                                   shape = (0,),
                                   dtype = value,
                                   maxshape = (None,))

        transform = sparseTensor.array_to_sparseTensor

        dl.genSampleLoadOrder()
        for evIndex in tqdm.tqdm(dl.sampleLoadOrder):
            hits, edep = dl.load_image(evIndex)
            evinfo = dl.evinfo_ev

            hitsST, edepST = transform([hits], [edep])
            output = net.forward(hitsST)
            inference = net.criterion.feature_map(output)
            
            inference_arr = np.empty(shape = (len(output),),
                                     dtype = output_dtypes['inference'])

            inference_arr['eventID'] = evIndex*np.ones(len(output), dtype = "u4")
            inference_arr['x'] = output.coordinates.detach().cpu().numpy()[:,1]
            inference_arr['y'] = output.coordinates.detach().cpu().numpy()[:,2]
            inference_arr['z'] = output.coordinates.detach().cpu().numpy()[:,3]

            inference_arr['dE'] = inference[0].detach().cpu().numpy()
            inference_arr['dE_err'] = inference[1].detach().cpu().numpy()

            write_to_output(outfile, hits, edep, inference_arr, evinfo)

        outfile.close()

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--manifest', type = str,
                        required = True,
                        help = "network manifest yaml file")
    parser.add_argument('-v', '--verbose',
                        action = 'store_true',
                        help = "print extra debug messages")
    parser.add_argument('-c', '--checkpoint', type = str,
                        required = True,
                        help = "checkpoint file to start from")
    parser.add_argument('-o', '--outdir', type = str,
                        required = True,
                        help = "output")
    
    args = parser.parse_args()

    main(args)