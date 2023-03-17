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
            f = h5py.File(fileName)
            eventIDs = f['evinfo']['eventID']
            
            nImages += len(np.unique(eventIDs))

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
        hits_ev = self.hits[hits_mask]

        edep_mask = self.edep['eventID'] == event_id
        edep_ev = self.edep[edep_mask]
        
        return hits_ev, edep_ev

