
import MinkowskiEngine as ME
ME.set_sparse_tensor_operation_mode(ME.SparseTensorOperationMode.SHARE_COORDINATE_MANAGER)

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import h5py

import numpy as np

from LarpixParser import hit_parser as HitParser
from LarpixParser import event_parser as EvtParser
from LarpixParser import util
from LarpixParser.geom_to_dict import larpix_layout_to_dict

from NDeventDisplay.voxelize import voxelize

from . import detector

class DataLoader:
    """
    This version of the DataLoader class is meant to parse the pared down
    data format.  It should be faster, since all but the needed information
    has been removed.
    """
    def __init__(self, infileList, batchSize = 10):
        self.fileList = infileList

        self.batchSize = batchSize

        nImages = 0
        for fileName in self.fileList:
            try:
                f = h5py.File(fileName)
                eventIDs = f['evinfo']['eventID']
            
                nImages += len(np.unique(eventIDs))
            except OSError:
                print ("Skipping bad file", fileName)
                self.fileList.remove(fileName)

        self.batchesPerEpoch = int(nImages/self.batchSize)
        
    def setFileLoadOrder(self):
        # set the order in which the files will be parsed
        # this should be redone at the beginning of every epoch
        nFiles = len(self.fileList)
        self.fileLoadOrder = np.random.choice(nFiles,
                                              size = nFiles,
                                              replace = False)
        
    def loadNextFile(self, fileIndex):
        # prime the next file.  This is done after the previous
        # file has been fully iterated through
        self.currentFileName = self.fileList[fileIndex]
        f = h5py.File(self.currentFileName)
        self.hits = f['hits']
        self.edep = f['edep']
        self.evinfo = f['evinfo']

        self.setSampleLoadOrder()
        
    def setSampleLoadOrder(self):
        # set the order that events/images within a given file
        # are sampled
        # This should be redone after each file is loaded
        # self.loadOrder = np.arange(self.t0_grp.shape[0])
        nImages = len(np.unique(self.evinfo['eventID']))
        self.sampleLoadOrder = np.random.choice(nImages,
                                                size = nImages,
                                                replace = False)

    def load(self):
        for fileIndex in self.fileLoadOrder:
            self.loadNextFile(fileIndex)
            hits = []
            edep = []
            for evtIndex in self.sampleLoadOrder:
                theseHits, theseEdep = self.load_event(evtIndex)
                if len(theseHits) == 0:
                    continue
                else:
                    hits.append(theseHits)
                    edep.append(theseEdep)

                if len(hits) == self.batchSize:
                    yield array_to_sparseTensor(hits, edep)
                    hits = []
                    edep = []
            
    def load_event(self, event_id):
        # load a given event from the currently loaded file

        hits_mask = self.hits['eventID'] == event_id
        self.hits_ev = self.hits[hits_mask]

        edep_mask = self.edep['eventID'] == event_id
        self.edep_ev = self.edep[edep_mask]

        return self.hits_ev, self.edep_ev

class RawDataLoader:
    """
    This version of the DataLoader class is meant to parse the raw
    larpix + voxels format.  It has much more information than is needed
    for 3D point cloud inference, so it is not ideal for training 
    """
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
        self.traj = f['trajectories']

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
                    if len(theseHits) == 0:
                        continue
                    elif theseHits[0].shape[0] == 0 or theseTracks[0].shape[0] == 0:
                        continue
                    else:
                        hits.append(theseHits)
                        tracks.append(theseTracks)
                else:
                    yield array_to_sparseTensor(hits, tracks)
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
        trackAssn_ev = self.assn[pckt_mask] 
        trackIDs_ev = trackAssn_ev['track_ids']
        trackFractions_ev = trackAssn_ev['fraction']
        # print (trackIDs_ev, trackFractions_ev)
        # print (trackIDs_ev.shape, trackFractions_ev.shape)
        # print (trackFractions_ev == np.max(trackFractions_ev, axis = -1))
        
        strongestTrack = trackIDs_ev[trackFractions_ev == np.max(trackFractions_ev, axis = -1)]
        # print (event_id, trackIDs_ev)
        # print (strongestTrack)
        # tracks_ev = self.tracks[trackIDs_ev]

        t0_correction = -38
        # print ("using t0 correction of", t0_correction)

        hitX, hitY, hitZ, dQ = HitParser.hit_parser_charge(t0 + t0_correction,
                                                           packets_ev,
                                                           self.geom_dict,
                                                           self.run_config,
                                                           drift_model = 2)
        
        vox_ev_id = np.unique(EvtParser.packet_to_eventid(self.assn,
                                                          self.tracks)[pckt_mask])
        if len(vox_ev_id) == 1:
            vox_mask = self.voxels['eventID'] == vox_ev_id
        else:
            try:
                vox_mask = np.logical_and(*[self.voxels['eventID'] == thisev_id
                                            for thisev_id in vox_ev_id])
            except ValueError:
                print ("empty event found at EVID", event_id)
                print ([self.voxels['eventID'] == thisev_id
                        for thisev_id in vox_ev_id])
                return np.array([]), np.array([])

        vox_ev = self.voxels[vox_mask]

        # HACK - force coordinate spacing to be ~1 by dividing by pixel pitch
        # figure out how to properly map pixels to integer coordinates
        # hits = (np.array(hitZ)/10/self.pixelPitch,
        #         np.array(hitX)/10/self.pixelPitch,
        #         np.array(hitY)/10/self.pixelPitch,
        #         np.array(dQ))
        hits = (np.array(hitZ)/10/self.pixelPitch,
                np.array(hitY)/10/self.pixelPitch,
                np.array(hitX)/10/self.pixelPitch,
                np.array(dQ))
        
        voxels = (vox_ev['xBin']/self.pixelPitch,
                  vox_ev['yBin']/self.pixelPitch,
                  vox_ev['zBin']/self.pixelPitch,
                  vox_ev['dE'])
        
        return hits, voxels

def array_to_sparseTensor(hitList, edepList):
    ME.clear_global_coordinate_manager()

    hitCoordTensors = []
    hitFeatureTensors = []
    
    edepCoordTensors = []
    edepFeatureTensors = []

    LarpixPadCoordTensors = []
    LarpixPadFeatureTensors = []

    EdepPadCoordTensors = []
    EdepPadFeatureTensors = []

    for hits, edep in zip(hitList, edepList):
        
        # trackX, trackZ, trackY, dE = edep
        edepX = edep['x']
        edepY = edep['z']
        edepZ = edep['y']
        dE = edep['dE']
        
        edepCoords = torch.FloatTensor(np.array([edepX, edepY, edepZ])).T
        edepFeature = torch.FloatTensor(np.array([dE])).T
                
        edepCoordTensors.append(edepCoords)
        edepFeatureTensors.append(edepFeature)

        # hitsX, hitsY, hitsZ, hitsQ = hits
        hitsX = hits['x']
        hitsY = hits['y']
        hitsZ = hits['z']
        hitsQ = hits['q']

        hitCoords = torch.FloatTensor(np.array([hitsX, hitsY, hitsZ])).T
        hitFeature = torch.FloatTensor(np.array([hitsQ])).T
            
        hitCoordTensors.append(hitCoords)
        hitFeatureTensors.append(hitFeature)

        LarpixPadCoords = edepCoords
        LarpixPadFeature = torch.zeros((LarpixPadCoords.shape[0], 1))
        
        LarpixPadCoordTensors.append(LarpixPadCoords)
        LarpixPadFeatureTensors.append(LarpixPadFeature)

        EdepPadCoords = hitCoords
        EdepPadFeature = torch.zeros((EdepPadCoords.shape[0], 1))
        
        EdepPadCoordTensors.append(EdepPadCoords)
        EdepPadFeatureTensors.append(EdepPadFeature)

            
    hitCoords, hitFeature = ME.utils.sparse_collate(hitCoordTensors, 
                                                    hitFeatureTensors,
                                                    dtype = torch.int32)
                
    edepCoords, edepFeature = ME.utils.sparse_collate(edepCoordTensors, 
                                                      edepFeatureTensors,
                                                      dtype = torch.int32)
    
    LarpixPadCoords, LarpixPadFeature = ME.utils.sparse_collate(LarpixPadCoordTensors, 
                                                                LarpixPadFeatureTensors,
                                                                dtype = torch.int32)

    EdepPadCoords, EdepPadFeature = ME.utils.sparse_collate(EdepPadCoordTensors, 
                                                            EdepPadFeatureTensors,
                                                            dtype = torch.int32)
                
    larpix = ME.SparseTensor(features = hitFeature.to(device),
                             coordinates = hitCoords.to(device))
    edep = ME.SparseTensor(features = edepFeature.to(device),
                           coordinates = edepCoords.to(device),
                           coordinate_manager = larpix.coordinate_manager,
                           )
    LarpixPad = ME.SparseTensor(features = LarpixPadFeature.to(device),
                                coordinate_map_key = edep.coordinate_map_key,
                                coordinate_manager = larpix.coordinate_manager,
                                )
    EdepPad = ME.SparseTensor(features = EdepPadFeature.to(device),
                              coordinate_map_key = larpix.coordinate_map_key,
                              coordinate_manager = larpix.coordinate_manager,
                              )

    # print (larpix.shape, edep.shape, LarpixPad.shape)
    # print ("larpix center", torch.mean(larpix.coordinates.float(), dim = 0))
    # print ("edep center", torch.mean(edep.coordinates.float(), dim = 0))
    larpix = larpix + LarpixPad
    edep = edep + EdepPad
    # print (larpix.shape, edep.shape)
    
    return larpix, edep
