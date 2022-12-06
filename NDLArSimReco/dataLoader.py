import h5py

import numpy as np

from LarpixParser import hit_parser as HitParser
from LarpixParser import event_parser as EvtParser
from LarpixParser import util
from LarpixParser.geom_to_dict import larpix_layout_to_dict

from . import detector

class DataLoader:
    def __init__(self, h5File):
        self.fileName = h5File
        f = h5py.File(self.fileName)
        self.packets = f['packets']
        self.tracks = f['tracks']
        self.assn = f['mc_packets_assn']

        self.t0_grp = EvtParser.get_t0(self.packets)
        self.geom_dict = larpix_layout_to_dict("multi_tile_layout-3.0.40",
                                               save_dict = False)

        self.run_config = util.get_run_config("ndlar-module.yaml",
                                              use_builtin = True)
    
    def sampleLoadOrder(self):
        # self.loadOrder = np.arange(self.t0_grp.shape[0])
        nBatches = self.t0_grp.shape[0]
        self.loadOrder = np.random.choice(nBatches,
                                          size = nBatches,
                                          replace = False)

    def load(self):
        for i in self.loadOrder:
            yield load_event(i)
        
    def load_event(self, event_id):
        t0 = self.t0_grp[event_id][0]
        print("--------event_id: ", event_id)
        print(self.run_config.keys())
        ti = t0 + self.run_config['time_interval'][0]/self.run_config['CLOCK_CYCLE']
        tf = t0 + self.run_config['time_interval'][1]/self.run_config['CLOCK_CYCLE']
        
        pckt_mask = (self.packets['timestamp'] > ti) & (self.packets['timestamp'] < tf)
        packets_ev = self.packets[pckt_mask]

        hitX, hitY, hitZ, dQ = HitParser.hit_parser_charge(t0,
                                                           packets_ev,
                                                           self.geom_dict,
                                                           self.run_config)

        track_ev_id = np.unique(EvtParser.packet_to_eventid(self.assn,
                                                            self.tracks)[pckt_mask])
        track_mask = self.tracks['eventID'] == track_ev_id
        tracks_ev = self.tracks[track_mask]

        track_xStart = tracks_ev['x_start']
        track_yStart = tracks_ev['y_start']
        track_zStart = tracks_ev['z_start']

        track_xEnd = tracks_ev['x_end']
        track_yEnd = tracks_ev['y_end']
        track_zEnd = tracks_ev['z_end']

        track_dE = tracks_ev['dE']

        hits = (np.array(hitZ)/10,
                np.array(hitX)/10,
                np.array(hitY)/10,
                np.array(dQ))
        
        tracks = (track_xStart, track_xEnd,
                  track_zStart, track_zEnd,
                  track_yStart, track_yEnd,
                  track_dE)

        return hits, tracks
