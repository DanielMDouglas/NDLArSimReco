# this script can join output files from proc.py
# this entails adjusting the eventID index
# and concatenating

import numpy as np
import h5py

from proc import output_dtypes

def main(args):
    outfile = h5py.File(args.outfile, 'w')

    for key, value in output_dtypes.items():
        outfile.create_dataset(key,
                               shape = (0,),
                               dtype = value,
                               maxshape = (None,))

    for infile in args.infileList:
        f = h5py.File(infile, 'r')
        # get previous number of events
        try:
            nEv_prev = np.max(outfile['evinfo']['eventID'])
        except ValueError:
            nEv_prev = 0

        print (nEv_prev)
            
        # increment the eventID in the input arrays
        # for key, value in output_dtypes.items():
            
        # resize outfile arrays
        for key in output_dtypes.keys():
            size_prev = len(outfile[key])
            size_thisfile = len(f[key])

            tmp_arr = np.empty(size_thisfile, dtype = output_dtypes[key])
            tmp_arr = f[key][:].copy()
            tmp_arr['eventID'] += nEv_prev

            outfile[key].resize((size_prev + size_thisfile,))
            # append input arrays
            outfile[key][size_prev:] = tmp_arr
            # outfile[key][size_prev:] = f[key]
            # outfile[key][size_prev:]['eventID'] += nEv_prev

        f.close()

    outfile.close()
        
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    # actually only makes sense to pass one at a time...
    parser.add_argument('infileList', type = str,
                        nargs = '+',
                        help = "input files to concatenate")
    parser.add_argument('-o', '--outfile', type = str,
                        default = "testoutConcat.h5",
                        help = "output")
    
    args = parser.parse_args()
    
    main(args)
