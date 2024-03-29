# this script performs the preprocessing needed for the input + ground truth images
# the voxelization is already handled by the module in larnd-sim (my fork)
# this script will pare away all of the un-needed information, leaving the
# 3D positions and features of the input hits and edep voxels

# proc (first) -> join (optional) -> cut (optional) -> shift_indices (last) 

import numpy as np
import h5py
import particle

from LarpixParser import hit_parser as HitParser
from LarpixParser import event_parser as EvtParser
from LarpixParser import util
from LarpixParser.geom_to_dict import larpix_layout_to_dict

from NDeventDisplay.voxelize import voxelize

from NDLArSimReco import detector
from NDLArSimReco.dataLoader import RawDataLoader

# file layout should be:

# hits:
# eventID x y z q

# edep voxels:
# eventID x y z dE

# primary PID:
# eventID PID

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
                                   ("dE", "f4"),
                                   ("PID", "u4"),
                                   ("semantic_label", "u4")],
                                  align = True),
                 "evinfo": np.dtype([("eventID", "u4"),
                                     ("primaryPID", "i4"),
                                     ("primaryE", "f4"),
                                     ("start_dEdx", "f4")],
                                    align = True),
                 }

def get_primary(dl, event_id):
    t0 = dl.t0_grp[event_id][0]

    ti = t0 + dl.run_config['time_interval'][0]/dl.run_config['CLOCK_CYCLE']
    tf = t0 + dl.run_config['time_interval'][1]/dl.run_config['CLOCK_CYCLE']

    pckt_mask = (dl.packets['timestamp'] > ti) & (dl.packets['timestamp'] < tf)
    packets_ev = dl.packets[pckt_mask]

    traj_ev_id = np.unique(EvtParser.packet_to_eventid(dl.assn,
                                                       dl.tracks)[pckt_mask])
    if len(traj_ev_id) == 1:
        traj_mask = dl.traj['eventID'] == traj_ev_id
    else:
        try:
            traj_mask = np.logical_and(*[dl.traj['eventID'] == thisev_id
                                        for thisev_id in traj_ev_id])
        except ValueError:
            print ("empty event found at EVID", event_id)
            print ([dl.traj['eventID'] == thisev_id
                    for thisev_id in traj_ev_id])
            return np.array([]), np.array([])

    traj_ev = dl.traj[traj_mask]

    primMask = traj_ev['parentID'] == -1
    return traj_ev[primMask]

def get_tracks(dl, event_id):
    t0 = dl.t0_grp[event_id][0]

    ti = t0 + dl.run_config['time_interval'][0]/dl.run_config['CLOCK_CYCLE']
    tf = t0 + dl.run_config['time_interval'][1]/dl.run_config['CLOCK_CYCLE']

    pckt_mask = (dl.packets['timestamp'] > ti) & (dl.packets['timestamp'] < tf)
    packets_ev = dl.packets[pckt_mask]

    tracks_ev_id = np.unique(EvtParser.packet_to_eventid(dl.assn,
                                                         dl.tracks)[pckt_mask])
    if len(tracks_ev_id) == 1:
        track_mask = dl.tracks['eventID'] == tracks_ev_id
    else:
        try:
            track_mask = np.logical_and(*[dl.tracks['eventID'] == thisev_id
                                         for thisev_id in tracks_ev_id])
        except ValueError:
            print ("empty event found at EVID", tracks_event_id)
            print ([dl.traj['eventID'] == thisev_id
                    for thisev_id in tracks_ev_id])
            return np.array([]), np.array([])

    tracks_ev = dl.tracks[track_mask]

    return tracks_ev

def get_most_upstream_edep_pos(tracks):
    firstTrackMask = tracks['segment_id'] == np.min(tracks['segment_id'])
    position = np.array([tracks['x_start'][firstTrackMask],
                         tracks['y_start'][firstTrackMask],
                         tracks['z_start'][firstTrackMask]])

    return position

def select_downstream_segments(tracks, pos, vector, distance):

    segPositions = np.array([tracks['x_start'] - pos[0],
                             tracks['y_start'] - pos[1],
                             tracks['z_start'] - pos[2]])
    axialDist = np.dot(vector, segPositions)
    downstreamMask = axialDist < distance
    inFrontMask = axialDist >= 0
    mask = downstreamMask*inFrontMask

    return tracks[mask]

def write_to_output(outfile, evHits, evEdep, evEv):
    nHits_prev = len(outfile['hits'])
    nHits_ev = len(evHits)
    outfile['hits'].resize((nHits_prev + nHits_ev,))
    outfile['hits'][nHits_prev:] = evHits
    
    nEdep_prev = len(outfile['edep'])
    nEdep_ev = len(evEdep)
    outfile['edep'].resize((nEdep_prev + nEdep_ev,))
    outfile['edep'][nEdep_prev:] = evEdep
            
    nEv_prev = len(outfile['evinfo'])
    nEv_ev = len(evEv)
    outfile['evinfo'].resize((nEv_prev + nEv_ev,))
    outfile['evinfo'][nEv_prev:] = evEv

def main(args):
    dl = RawDataLoader([args.infileList], detectorConfig = args.config)
    outfile = h5py.File(args.outfile, 'w')

    for key, value in output_dtypes.items():
        outfile.create_dataset(key,
                               shape = (0,),
                               dtype = value,
                               maxshape = (None,))

    dl.loadNextFile(0)
    nEvents = dl.t0_grp.shape[0]

    for event_id in range(nEvents):
        hits, voxels = dl.load_image(event_id)
        prim = get_primary(dl, event_id)
        evTrack = get_tracks(dl, event_id)
        primPID = prim['pdgId']

        primM = np.array([particle.Particle.from_pdgid(pid).mass
                          for pid in primPID])
        primP2 = np.sum(np.power(prim['pxyz_start'], 2), axis = 1)
        primE = np.sqrt(primP2 + np.power(primM, 2))

        print ("mom", prim['pxyz_start'])
        momentum = [prim['pxyz_start'][0,2]/10,
                    prim['pxyz_start'][0,1]/10,
                    prim['pxyz_start'][0,0]/10]
        momentumDir = momentum/np.linalg.norm(momentum)

        upstreamPos = get_most_upstream_edep_pos(evTrack)

        distance = 3.
        startTrack = select_downstream_segments(evTrack, upstreamPos, momentumDir, distance)
        
        start_dEdx = np.sum(startTrack['dE'])/distance

        nHits_ev = len(hits[0])
        evHits = np.empty(nHits_ev, dtype = output_dtypes['hits'])
        evHits['eventID'] = event_id*np.ones(nHits_ev)

        # switch x and z coordinates for hits
        # we're using the ND coordinates, so
        # x is the horizontal (beam-left) direction
        # z is the beam direction
        evHits['z'] = np.array(hits[0])
        evHits['y'] = np.array(hits[1])
        evHits['x'] = np.array(hits[2])
        evHits['q'] = np.array(hits[3])

        edepEthreshold = 0.25
        thresholdMask = (np.array(voxels[3]) > edepEthreshold)

        nEdep_ev = int(sum(thresholdMask))
        evEdep = np.empty(nEdep_ev, dtype = output_dtypes['edep'])
        evEdep['eventID'] = event_id*np.ones(nEdep_ev)
        evEdep['x'] = np.array(voxels[0])[thresholdMask]
        evEdep['y'] = np.array(voxels[1])[thresholdMask]
        evEdep['z'] = np.array(voxels[2])[thresholdMask]
        evEdep['dE'] = np.array(voxels[3])[thresholdMask]
        evEdep['PID'] = np.array(voxels[4])[thresholdMask]
        evEdep['semantic_label'] = np.array(voxels[5])[thresholdMask]

        nEv_ev = len(primPID)
        evEv = np.empty(nEv_ev, dtype = output_dtypes['evinfo'])
        evEv['eventID'] = event_id*np.ones(nEv_ev)
        evEv['primaryPID'] = np.array(primPID)
        evEv['primaryE'] = np.array(primE)
        evEv['start_dEdx'] = np.array(start_dEdx)

        print (event_id)
        print ("nHits", nHits_ev)
        print ("nEdep", nEdep_ev)
        print ("nEv", nEv_ev)

        # if (nHits_ev > 0) and (nEdep_ev > 0) and (nEv_ev == 1):
        if (nHits_ev > 0) and (nEdep_ev > 0):
            print ("writing")
            write_to_output(outfile, evHits, evEdep, evEv)

        else:
            print ("skipping")

    outfile.close()
        
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    # actually only makes sense to pass one at a time...
    parser.add_argument('-i', '--infileList', type = str,
                        default = "/home/dan/studies/NDLArSimReco/NDLArSimReco/manifests/localTestManifest.yaml",
                        help = "input")
    parser.add_argument('-s', '--swapaxes',
                        action = 'store_true',
                        help = 'whether or not to swap the x and z axes of the reconstructed hit positions.  By default, this is true.  Including this flag will skip this step and assume that the edep-sim segments are already in the larpix coordinate system.')
    parser.add_argument('-c', '--config', type = str,
                        default = "nd-lar",
                        help = "detector config (default: nd-lar)")
    parser.add_argument('-o', '--outfile', type = str,
                        default = "testout.h5",
                        help = "output hdf5")

    args = parser.parse_args()
    
    main(args)
