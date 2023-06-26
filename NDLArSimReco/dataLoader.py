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

class GenericDataLoader:
    def __init__(self, infileList, batchSize = 10, sequentialLoad = False):
        self.fileList = infileList
        self.batchSize = batchSize
        self.sequential = sequentialLoad

        # This is a bit specific...
        # How to generically find the number of images?
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

        self.fileLoadOrder = np.empty(0,)
        self.sampleLoadOrder = np.empty(0,)

    def genFileLoadOrder(self):
        # set the order in which the files will be parsed
        # this should be redone at the beginning of every epoch
        nFiles = len(self.fileList)
        if self.sequential:
            self.fileLoadOrder = np.arange(nFiles)
        else:
            self.fileLoadOrder = np.random.choice(nFiles,
                                                  size = nFiles,
                                                  replace = False)
    def setFileLoadOrder(self, fileLoadOrder):
        """
        Setter method for the file load order
        """
        self.fileLoadOrder = np.array(fileLoadOrder)
        
    def getFileLoadOrder(self):
        """
        Getter method for the file load order
        """
        return self.fileLoadOrder.tolist()
    
    def loadNextFile(self, fileIndex):
        # prime the next file.  This is done after the previous
        # file has been fully iterated through
        self.currentFileName = self.fileList[fileIndex]
        self.currentFile = h5py.File(self.currentFileName)

    def setSampleLoadOrder(self, sampleLoadOrder):
        """
        Manually specify the sample load order from a list
        used when resuming a training process from a checkpoint
        """
        self.sampleLoadOrder = np.array(sampleLoadOrder)

    def getSampleLoadOrder(self):
        """
        Getter method for the sample load order
        used when saving a training checkpoint
        """
        return self.sampleLoadOrder.tolist()
        
    def genSampleLoadOrder(self):
        # set the order that events/images within a given file
        # are sampled
        # This should be redone after each file is loaded
        # self.loadOrder = np.arange(self.t0_grp.shape[0])
        nImages = len(np.unique(self.currentFile['evinfo']['eventID']))
        if self.sequential:
            self.sampleLoadOrder = np.arange(nImages)
        else:
            self.sampleLoadOrder = np.random.choice(nImages,
                                                    size = nImages,
                                                    replace = False)

    def load(self, transform = None):
        if len(self.fileLoadOrder) == 0: 
            self.genFileLoadOrder()
        for fileIndex in self.fileLoadOrder:
            self.loadNextFile(fileIndex)
            if len(self.sampleLoadOrder) == 0: 
                self.genSampleLoadOrder()
            inputs = []
            truths = []
            for imgIndex in self.sampleLoadOrder:
                theseInpts, theseTruths = self.load_image(imgIndex)
                if len(theseInpts) == 0:
                    continue
                else:
                    inputs.append(theseInpts)
                    truths.append(theseTruths)

                if len(inputs) == self.batchSize:
                    if transform:
                        yield transform(inputs, truths)
                    else:
                        yield inputs, truths
                    inputs = []
                    truths = []
            self.sampleLoadOrder = np.empty(0,)
        self.fileLoadOrder = np.empty(0,)

    def load_image(imgIndex):
        return None, None

class DataLoader (GenericDataLoader):
    """
    This version of the DataLoader class is meant to parse the pared down
    data format.  It should be faster, since all but the needed information
    has been removed.
    """            
    def load_image(self, eventIndex):
        # load a given event from the currently loaded file
        event_id = np.unique(self.currentFile['evinfo']['eventID'])[eventIndex]

        hits_mask = self.currentFile['hits']['eventID'] == event_id
        self.hits_ev = self.currentFile['hits'][hits_mask]

        edep_mask = self.currentFile['edep']['eventID'] == event_id
        self.edep_ev = self.currentFile['edep'][edep_mask]

        evinfo_mask = self.currentFile['evinfo']['eventID'] == event_id
        self.evinfo_ev = self.currentFile['evinfo'][evinfo_mask]
        
        return self.hits_ev, self.edep_ev

class DataLoaderWithEvinfo (GenericDataLoader):
    """
    This version of the DataLoader class is meant to parse the pared down
    data format.  It should be faster, since all but the needed information
    has been removed.
    """            
    def load(self, transform = None):
        for fileIndex in self.fileLoadOrder:
            self.loadNextFile(fileIndex)
            if len(self.sampleLoadOrder) == 0: 
                self.genSampleLoadOrder()
            hits = []
            edeps = []
            evinfos = []
            for imgIndex in self.sampleLoadOrder:
                theseHits, theseEdeps, theseEvinfos = self.load_image(imgIndex)
                if len(theseHits) == 0:
                    continue
                else:
                    hits.append(theseHits)
                    edeps.append(theseEdeps)
                    evinfos.append(theseEvinfos)

                if len(hits) == self.batchSize:
                    if transform:
                        yield transform(hits, edeps, evinfos)
                    else:
                        yield hits, edeps, evinfos
                    hits = []
                    edeps = []
                    evinfos = []
            self.sampleLoadOrder = np.empty(0,)
    def load_image(self, eventIndex):
        # load a given event from the currently loaded file
        event_id = np.unique(self.currentFile['evinfo']['eventID'])[eventIndex]

        hits_mask = self.currentFile['hits']['eventID'] == event_id
        self.hits_ev = self.currentFile['hits'][hits_mask]

        edep_mask = self.currentFile['edep']['eventID'] == event_id
        self.edep_ev = self.currentFile['edep'][edep_mask]

        evinfo_mask = self.currentFile['evinfo']['eventID'] == event_id
        self.evinfo_ev = self.currentFile['evinfo'][evinfo_mask]
        
        return self.hits_ev, self.edep_ev, self.evinfo_ev

class ClassifierDataLoader (GenericDataLoader):
    """
    This instance of the DataLoader class is mean for training a classifier
    network.  It should yield inferred edep-sim images (possibly G.T. images
    as well), alongside the true primary particle type
    """
    def load(self, transform = None):
        if len(self.fileLoadOrder) == 0: 
            self.genFileLoadOrder()
        for fileIndex in self.fileLoadOrder:
            self.loadNextFile(fileIndex)
            if len(self.sampleLoadOrder) == 0: 
                self.genSampleLoadOrder()
            inputs = []
            truths = []
            for imgIndex in self.sampleLoadOrder:
                theseInpts, theseTruths = self.load_image(imgIndex)
                if len(theseInpts) == 0:
                    continue
                else:
                    inputs.append(theseInpts)
                    truths.append(theseTruths)

                if len(inputs) == self.batchSize:
                    if transform:
                        yield transform(inputs, truths)
                    else:
                        yield inputs, truths
                    inputs = []
                    truths = []
            self.sampleLoadOrder = np.empty(0,)
        self.fileLoadOrder = np.empty(0,)

    def load_image(self, eventIndex):
        # load a given event from the currently loaded file
        event_id = np.unique(self.currentFile['evinfo']['eventID'])[eventIndex]

        inference_mask = self.currentFile['inference']['eventID'] == event_id
        self.inference_ev = self.currentFile['inference'][inference_mask]

        evinfo_mask = self.currentFile['evinfo']['eventID'] == event_id
        self.evinfo_ev = self.currentFile['evinfo'][evinfo_mask]
                                          
        return self.inference_ev, self.evinfo_ev


class ClassifierDataLoaderGT (GenericDataLoader):
    """
    This instance of the DataLoader class is mean for training a classifier
    network.  It should yield inferred edep-sim images (possibly G.T. images
    as well), alongside the true primary particle type
    """
    def load_image(self, eventIndex):
        # load a given event from the currently loaded file
        event_id = np.unique(self.currentFile['evinfo']['eventID'])[eventIndex]

        edep_mask = self.currentFile['edep']['eventID'] == event_id
        self.edep_ev = self.currentFile['edep'][edep_mask]

        evinfo_mask = self.currentFile['evinfo']['eventID'] == event_id
        self.evinfo_ev = self.currentFile['evinfo'][evinfo_mask]
                                          
        return self.edep_ev, self.evinfo_ev

class ClassifierDataLoaderLNDSM (GenericDataLoader):
    """
    This instance of the DataLoader class is mean for training a classifier
    network.  It should yield larnd-sim images, alongside the true primary particle type
    """
    def load_image(self, eventIndex):
        # load a given event from the currently loaded file
        event_id = np.unique(self.currentFile['evinfo']['eventID'])[eventIndex]

        hits_mask = self.currentFile['hits']['eventID'] == event_id
        self.hits_ev = self.currentFile['hits'][hits_mask]

        evinfo_mask = self.currentFile['evinfo']['eventID'] == event_id
        self.evinfo_ev = self.currentFile['evinfo'][evinfo_mask]

        return self.hits_ev, self.evinfo_ev

        
class RawDataLoader (GenericDataLoader):
    """
    This version of the DataLoader class is meant to parse the raw
    larpix + voxels format.  It has much more information than is needed
    for 3D point cloud inference, so it is not ideal for training 
    """
    def __init__(self, infileList, batchSize = 10, detectorConfig = "nd-lar"):
        print (detectorConfig)
        self.fileList = infileList

        print ("building geometry lookup...")
        if detectorConfig == 'nd-lar':
            self.geom_dict = larpix_layout_to_dict("multi_tile_layout-3.0.40",
                                                   save_dict = False)
            self.switch_xz = True
        elif detectorConfig == '2x2':
            self.geom_dict = larpix_layout_to_dict("multi_tile_layout-2.3.16",
                                                   save_dict = False)
            self.switch_xz = False
        
        print ("building run configuration...")
        if detectorConfig == 'nd-lar':
            self.run_config = util.get_run_config("ndlar-module.yaml",
                                                  use_builtin = True)
        elif detectorConfig == '2x2':
            self.run_config = util.get_run_config("2x2.yaml",
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

        self.genSampleLoadOrder()
        
    def genSampleLoadOrder(self):
        # set the order that events/images within a given file
        # are sampled
        # This should be redone after each file is loaded
        # self.loadOrder = np.arange(self.t0_grp.shape[0])
        nBatches = self.t0_grp.shape[0]
        self.sampleLoadOrder = np.random.choice(nBatches,
                                                size = nBatches,
                                                replace = False)
        print ("generating a new sample load order!")

    def load_image(self, event_id, get_true_tracks = False):
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

        strongestTrack = np.empty((packets_ev.shape), dtype = self.tracks.dtype)
        assn = []
        for i, (packetTID, packetFraction) in enumerate(zip(trackIDs_ev, trackFractions_ev)):
            maxInd = list(packetFraction).index(max(packetFraction))
            trackInd = packetTID[maxInd]
            strongestTrack[i] = self.tracks[trackInd]
            
        # it should be similar to how I do it for the other thing
        # an array with a row for each hit
        # and an entry with the track index
            
        # strongestTrack = trackIDs_ev[trackFractions_ev == np.max(trackFractions_ev, axis = -1)]
        # tracks_ev = self.tracks[trackIDs_ev]

        t0_correction = -38

        hitX, hitY, hitZ, dQ = HitParser.hit_parser_charge(t0 + t0_correction,
                                                           packets_ev,
                                                           self.geom_dict,
                                                           self.run_config,
                                                           drift_model = 2,
                                                           switch_xz = self.switch_xz)
        hits_ev = np.array([hitX, hitY, hitZ, dQ])
        
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
        hits = (np.array(hitX)/10/self.pixelPitch,
                np.array(hitY)/10/self.pixelPitch,
                np.array(hitZ)/10/self.pixelPitch,
                np.array(dQ))
        
        voxels = (vox_ev['xBin']/self.pixelPitch,
                  vox_ev['yBin']/self.pixelPitch,
                  vox_ev['zBin']/self.pixelPitch,
                  vox_ev['dE'])

        if get_true_tracks:
            tracks = None
            return hits, voxels, strongestTrack, hits_ev
        else:
            return hits, voxels

class DataLoaderFactoryClass:
    map =  {'DataLoader': DataLoader,
            'ClassifierDataLoader': ClassifierDataLoader,
            'ClassifierDataLoaderGT': ClassifierDataLoaderGT,
            'ClassifierDataLoaderLNDSM': ClassifierDataLoaderLNDSM,
            'RawDataLoader': RawDataLoader,
            'DataLoaderWithEvinfo': DataLoaderWithEvinfo,
            }
    def __getitem__(self, req):
        if req in self.map:
            return self.map[req]
        else:
            return DataLoader
dataLoaderFactory = DataLoaderFactoryClass()
