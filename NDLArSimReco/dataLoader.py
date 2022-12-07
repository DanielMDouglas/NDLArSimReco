import h5py

import numpy as np

from LarpixParser import hit_parser as HitParser
from LarpixParser import event_parser as EvtParser
from LarpixParser import util
from LarpixParser.geom_to_dict import larpix_layout_to_dict

from . import detector

class DataLoader:
    def __init__(self, infileList):
        self.fileList = infileList

        self.geom_dict = larpix_layout_to_dict("multi_tile_layout-3.0.40",
                                               save_dict = False)

        self.run_config = util.get_run_config("ndlar-module.yaml",
                                              use_builtin = True)

        xMin, xMax, xWidth = 425.0, 905.0, 0.38
        yMin, yMax, yWidth = -290.0, 390.0, 0.38
        zMin, zMax, zWidth = -210.0, 70.0, 0.38

        nVoxX = int((xMax - xMin)/xWidth)
        nVoxY = int((yMax - yMin)/yWidth)
        nVoxZ = int((zMax - zMin)/zWidth)

        self.trackVoxelEdges = (np.linspace(xMin, xMax, nVoxX + 1),
                                np.linspace(yMin, yMax, nVoxY + 1),
                                np.linspace(zMin, zMax, nVoxZ + 1))


    def trackVoxelizer(self, hits, tracks):
        (track_xStart, track_xEnd,
         track_yStart, track_yEnd,
         track_zStart, track_zEnd,
         track_dE) = tracks
        xMid = 0.5*(track_xStart + track_xEnd)
        yMid = 0.5*(track_yStart + track_yEnd)
        zMid = 0.5*(track_zStart + track_zEnd)
        (hitX,
         hitY,
         hitZ,
         dQ) = hits
        
        print ('x')
        print ("track", np.min(xMid), np.max(xMid))
        print ("hit", np.min(hitX), np.max(hitX))

        print ('y')
        print ("track", np.min(yMid), np.max(yMid))
        print ("hit", np.min(hitY), np.max(hitY))

        print ('z')
        print ("track", np.min(zMid), np.max(zMid))
        print ("hit", np.min(hitZ), np.max(hitZ))
        
    def setFileLoadOrder(self):
        # set the order in which the files will be parsed
        # this should be redone at the beginning of every epoch
        self.fileLoadOrder = np.random.choice(len(self.fileList),
                                              size = len(self.fileList),
                                              replace = False)
        
    def loadNextFile(self, fileIndex):
        # prime the next file.  This is done after the previous
        # file has been fully iterated through
        self.currentFileName = self.fileList[fileIndex]
        f = h5py.File(self.currentFileName)
        self.packets = f['packets']
        self.tracks = f['tracks']
        self.assn = f['mc_packets_assn']

        self.t0_grp = EvtParser.get_t0(self.packets)

        self.setSampleLoadOrder()
        
    def setSampleLoadOrder(self):
        # set the order that events/images within a given file
        # are sampled
        # This should be redone after each file is loaded
        # self.loadOrder = np.arange(self.t0_grp.shape[0])
        nBatches = self.t0_grp.shape[0]
        self.sampleLoadOrder = np.random.choice(nBatches,
                                                size = nBatches,
                                                replace = False)

    def load(self):
        for fileIndex in self.fileLoadOrder:
            self.loadNextFile(fileIndex)
            for evtIndex in self.sampleLoadOrder:
                yield self.load_event(evtIndex)
        
    def load_event(self, event_id):
        # load a given event from the currently loaded file
        t0 = self.t0_grp[event_id][0]
        # print("--------event_id: ", event_id)
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

        self.trackVoxelizer(hits, tracks)

        return hits, tracks
