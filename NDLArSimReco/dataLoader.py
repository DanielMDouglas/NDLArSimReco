import h5py

import numpy as np

from LarpixParser import hit_parser as HitParser
from LarpixParser import event_parser as EvtParser
from LarpixParser import util
from LarpixParser.geom_to_dict import larpix_layout_to_dict

from NDeventDisplay.voxelize import voxelize

from . import detector

class DataLoader:
    def __init__(self, infileList, batchSize = 10):
        self.fileList = infileList

        print ("building geometry lookup...")
        self.geom_dict = larpix_layout_to_dict("multi_tile_layout-3.0.40",
                                               save_dict = False)

        print ("building run configuration...")
        self.run_config = util.get_run_config("ndlar-module.yaml",
                                              use_builtin = True)

        self.pixelPitch = 0.38
        
        xMin, xMax, xWidth = 425.0, 905.0, self.pixelPitch
        yMin, yMax, yWidth = -290.0, 390.0, self.pixelPitch
        zMin, zMax, zWidth = -210.0, 70.0, self.pixelPitch

        nVoxX = int((xMax - xMin)/xWidth)
        nVoxY = int((yMax - yMin)/yWidth)
        nVoxZ = int((zMax - zMin)/zWidth)

        self.trackVoxelEdges = (np.linspace(xMin, xMax, nVoxX + 1),
                                np.linspace(yMin, yMax, nVoxY + 1),
                                np.linspace(zMin, zMax, nVoxZ + 1))

        self.batchSize = batchSize

        nImages = 0
        for fileName in self.fileList:
            f = h5py.File(fileName)
            packets = f['packets']
            t0_grp = EvtParser.get_t0(packets)

            nImages += len(np.unique(t0_grp))

        self.batchesPerEpoch = int(nImages/batchSize)
        
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
        self.voxels = f['track_voxels']
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
            hits = []
            tracks = []
            for evtIndex in self.sampleLoadOrder:
                if not len(hits) == self.batchSize:
                    theseHits, theseTracks = self.load_event(evtIndex)
                    if theseHits[0].shape[0] == 0 or theseTracks[0].shape[0] == 0:
                        continue
                    else:
                        hits.append(theseHits)
                        tracks.append(theseTracks)
                else:
                    yield hits, tracks
                    hits = []
                    tracks = []
            
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

        vox_ev_id = np.unique(EvtParser.packet_to_eventid(self.assn,
                                                          self.tracks)[pckt_mask])
        if len(vox_ev_id) == 1:
            vox_mask = self.voxels['eventID'] == vox_ev_id
        else:
            vox_mask = np.logical_and(*[self.voxels['eventID'] == thisev_id
                                          for thisev_id in vox_ev_id])
        vox_ev = self.voxels[vox_mask]

        # HACK - force coordinate spacing to be ~1 by dividing by pixel pitch
        # figure out how to properly map pixels to integer coordinates
        hits = (np.array(hitZ)/10/self.pixelPitch,
                np.array(hitX)/10/self.pixelPitch,
                np.array(hitY)/10/self.pixelPitch,
                np.array(dQ))

        voxels = (vox_ev['xBin']/self.pixelPitch,
                  vox_ev['yBin']/self.pixelPitch,
                  vox_ev['zBin']/self.pixelPitch,
                  vox_ev['dE'])
        
        return hits, voxels
