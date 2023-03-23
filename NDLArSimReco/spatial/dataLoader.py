import MinkowskiEngine as ME

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import h5py

import numpy as np

from LarpixParser import hit_parser as HitParser
from LarpixParser import event_parser as EvtParser
from LarpixParser import util
from LarpixParser.geom_to_dict import larpix_layout_to_dict

from NDeventDisplay.voxelize import voxelize

from NDLArSimReco import detector

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
            
            nImages += len(f['hits'])

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
        self.edep = f['edep']
        self.hits = f['hits']

        self.setSampleLoadOrder()
        
    def setSampleLoadOrder(self):
        # set the order that events/images within a given file
        # are sampled
        # This should be redone after each file is loaded
        # self.loadOrder = np.arange(self.t0_grp.shape[0])
        nImages = len(self.hits[:])
        self.sampleLoadOrder = np.random.choice(nImages,
                                                size = nImages,
                                                replace = False)

    def load(self):
        for fileIndex in self.fileLoadOrder:
            print ("loading next file")
            self.loadNextFile(fileIndex)
            hits = []
            edep = []
            for evtIndex in self.sampleLoadOrder:
                theseHits, theseEdep = self.load_event(evtIndex)
                l2 = np.power(theseHits[0] - theseEdep[0], 2) + \
                    np.power(theseHits[1] - theseEdep[1], 2) + \
                    np.power(theseHits[2] - theseEdep[2], 2)
                if len(theseHits) == 0:
                    continue
                elif l2 > 1.e2:
                    continue
                else: 
                    hits.append(theseHits)
                    edep.append(theseEdep)

                if len(hits) == self.batchSize:
                    yield array_to_tensor(hits, edep)
                    hits = []
                    edep = []
            
    def load_event(self, event_id):
        # load a given event from the currently loaded file

        hits_ev = self.hits[event_id]

        edep_ev = self.edep[event_id]
        
        return hits_ev, edep_ev

def array_to_tensor(hitList, edepList):
    # ME.clear_global_coordinate_manager()

    hitCoordTensors = []
    
    edepCoordTensors = []

    for hits, edep in zip(hitList, edepList):
        
        # trackX, trackZ, trackY, dE = edep
        edepX = edep[0]
        edepY = edep[1]
        edepZ = edep[2]
        
        edepCoords = torch.FloatTensor(np.array([edepX, edepY, edepZ])).T
                
        edepCoordTensors.append(edepCoords)

        # hitsX, hitsY, hitsZ, hitsQ = hits
        hitsX = hits[0]
        hitsY = hits[1]
        hitsZ = hits[2]

        hitCoords = torch.FloatTensor(np.array([hitsX, hitsY, hitsZ])).T
            
        hitCoordTensors.append(hitCoords)

    hitCoordTensors = torch.stack(hitCoordTensors).to(device)
    edepCoordTensors = torch.stack(edepCoordTensors).to(device)

    return hitCoordTensors, edepCoordTensors
