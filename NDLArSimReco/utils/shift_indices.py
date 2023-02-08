# this script shifts eventIDs down to their proper index
# i.e., there are no gaps in the resulting arrays
# [0, 1, 4, 5, 7] -> [0, 1, 2, 3, 4]

# proc (first) -> join (optional) -> cut (optional) -> shift_indices (last) 

import numpy as np
import h5py

from proc import output_dtypes

def main(args):
    outfile = h5py.File(args.outfile, 'w')
    
    f = h5py.File(args.infile, 'r')
        
    oldIDs = list(np.unique(f['evinfo']['eventID']))
    nEvents = len(oldIDs) 

    for key, value in output_dtypes.items():
        outfile.create_dataset(key,
                               shape = (0,),
                               dtype = value,
                               maxshape = (None,))


        size_prev = len(outfile[key])
        size_thisfile = len(f[key])

        tmp_arr = np.empty(size_thisfile, dtype = output_dtypes[key])
        tmp_arr = f[key][:].copy()

        tmp_arr['eventID'] = np.array([oldIDs.index(oldID) for oldID in f[key]['eventID']])

        outfile[key].resize((size_prev + size_thisfile,))

        outfile[key][size_prev:] = tmp_arr

    f.close()

    outfile.close()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--infile', type = str,
                        default = "/home/dan/studies/NDLArSimReco/NDLArSimReco/samples/edep_single_particle_larndsim_08364217-5952-4e55-814d-174ca6611bda_pared.h5",
                        help = "input")
    parser.add_argument('-o', '--outfile', type = str,
                        default = "testoutShift.h5",
                        help = "output")
    
    args = parser.parse_args()

    main(args)
