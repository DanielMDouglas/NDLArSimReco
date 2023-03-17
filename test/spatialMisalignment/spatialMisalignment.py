import numpy as np
import matplotlib.pyplot as plt

import h5py

from LarpixParser import hit_parser as HitParser
from LarpixParser import event_parser as EvtParser
from LarpixParser.geom_to_dict import larpix_layout_to_dict
from LarpixParser import util

import tqdm

def main(args):
    f = h5py.File(args.infile)

    tracks = f['tracks']

    packets = f['packets']

    assn = f['mc_packets_assn']

    geom_dict = larpix_layout_to_dict('multi_tile_layout-3.0.40',
                                      save_dict = False)
    run_config = util.get_run_config('ndlar-module.yaml',
                                     use_builtin = True)

    t0_grp = EvtParser.get_t0(packets)

    input_position = []
    hit_position = []
    
    for t0_row in tqdm.tqdm(t0_grp):

        t0 = t0_row[0]
        ti = t0 + run_config['time_interval'][0]/run_config['CLOCK_CYCLE']
        tf = t0 + run_config['time_interval'][1]/run_config['CLOCK_CYCLE']
        # print (t0)

        packet_mask = (packets['timestamp'] >= ti) & (packets['timestamp'] < tf)
        packets_ev = packets[packet_mask]

        trackEvID = max(EvtParser.packet_to_eventid(assn, tracks)[packet_mask])

        thisTrack = tracks[tracks['eventID'] == trackEvID]
        
        if thisTrack:
            thisTrackPos = 10*np.array([thisTrack['z'],
                                        thisTrack['y'],
                                        thisTrack['x'],
                                        ]).T[0]
        
            hitX,hitY,hitZ,hitQ = HitParser.hit_parser_charge(t0,
                                                              packets_ev,
                                                              geom_dict,
                                                              run_config,
                                                              drift_model = 2)
            thisHitPos = np.array([hitX, hitY, hitZ]).T
            
            input_position.append(thisTrackPos)
            hit_position.append(np.mean(thisHitPos, axis = 0))
            # print (thisTrackPos)
            # print (np.mean(thisHitPos, axis = 0))
            
            # print (input_position)
            # print (hit_position)
        
    input_position = np.array(input_position)
    hit_position = np.array(hit_position)
    difference = hit_position - input_position

    pos_map_dtype = np.dtype([('edep', 'f4'),
                              ('hits', 'f4'),
                              ('diff', 'f4')])
    
    with h5py.File(args.outfile, 'w') as of:
        of['edep'] = input_position
        of['hits'] = hit_position
        of['diff'] = difference
        
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('infile', type = str,
                        help = "input larnd-sim output file with a specific test pattern")
    parser.add_argument('outfile', type = str,
                        help = "output hdf5 file to save the results")
    args = parser.parse_args()

    main(args)
