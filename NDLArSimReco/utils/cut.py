# this script performs a cut on a particular particle type

# proc (first) -> join (optional) -> cut (optional) -> shift_indices (last) 

import numpy as np
import h5py

from proc import output_dtypes

def main(args):
    outfile = h5py.File(args.outfile, 'w')

    f = h5py.File(args.infile, 'r')

    primaryMask = (f['evinfo']['primaryPID'] == args.PID)
    goodEventIDs = f['evinfo']['eventID'][primaryMask]
    # get events that have the right PID
    # get the eventIDs
    
    # for each key:
    for key, value in output_dtypes.items():
        outfile.create_dataset(key,
                               shape = (0,),
                               dtype = value,
                               maxshape = (None,))


        size_thisfile = len(f[key])

        primaryMask = np.array([thisEvID in goodEventIDs for thisEvID in f[key]['eventID']])
        
        tmp_arr = np.empty(size_thisfile, dtype = output_dtypes[key])
        tmp_arr = f[key][:].copy()[primaryMask]

        outfile[key].resize((len(tmp_arr),))
        outfile[key][:] = tmp_arr
    # make a mask that's true if the eventID is in the good ones
    # save to tmp_arr
    # write to outfile
    
    f.close()

    outfile.close()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--infile', type = str,
                        default = "/home/dan/studies/NDLArSimReco/NDLArSimReco/samples/edep_single_particle_larndsim_08364217-5952-4e55-814d-174ca6611bda_pared.h5",
                        help = "input")
    parser.add_argument('-p', '--PID', type = int,
                        default = 13,
                        help = "input")
    parser.add_argument('-o', '--outfile', type = str,
                        default = "testoutCut.h5",
                        help = "output")
    
    args = parser.parse_args()
    
    main(args)

