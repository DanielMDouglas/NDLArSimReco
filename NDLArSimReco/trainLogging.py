import os
import yaml

import numpy as np
import torch

def loadManifestDict(manifest):
    """
    Load the manifest dictionary from a dictionary or a yaml path
    """
    if type(manifest) == dict: # if arg is a dict, do nothing
        manifestDict = manifest
    elif type(manifest) == str: # if arg is a yaml path, load
        with open(manifest, 'r') as mf:
            manifestDict = yaml.load(mf, Loader = yaml.FullLoader)
            
    assert type(manifestDict) == dict

    return manifestDict

def check_equality(entryA, entryB):
    """
    Check if two log entries correspond to the same log file.
    Log entries can be passed by file path or by LogEntry instance
    This can probably be simplified...
    """

    equality = False
    if type(entryA) == LogEntry and type(entryB) == LogEntry:
        equality = (os.path.abspath(entryA.outDir) == os.path.abspath(entryB.outDir))
    elif type(entryA) == str and type(entryB) == LogEntry:
        equality = (os.path.abspath(entryA) == os.path.abspath(entryB.outDir))
    elif type(entryA) == LogEntry and type(entryB) == str:
        equality = (os.path.abspath(entryA.outDir) == os.path.abspath(entryB))
    elif type(entryA) == str and type(entryB) == str:
        equality = (os.path.abspath(entryA) == os.path.abspath(entryB))

    return equality

class LogManager:
    def __init__(self, network):
        self.network = network
        self.outDir= network.outDir

        self.dataLoader = None # initially None, should be assigned when a dataLoader is used

        self.entries = []
        for existingCheckpoint in os.listdir(os.path.join(self.outDir,
                                                          "checkpoints")):
            logDir = os.path.join(self.outDir,
                                  "checkpoints",
                                  existingCheckpoint)

            self.entries.append(LogEntry(self, logDir))

        # sort by the epoch + 1.e-5*iter
        # WARNING: may break if an epoch has > 1.e5 batches
        self.entries.sort(key = lambda x: int(x.outDir.split('_')[-2]) + 1.e-5*int(x.outDir.split('_')[-1]))

        self.lossBuffer = []
            
    def log_state(self):
        """
        Create a new logEntry and write it to disk
        """

        n_epoch = self.network.n_epoch
        n_iter = self.network.n_iter

        checkpointName = 'checkpoint_'+str(n_epoch)+'_'+str(n_iter)
        logDir = os.path.join(self.outDir,
                              "checkpoints",
                              checkpointName)
        if not os.path.exists(logDir):
            os.mkdir(logDir)

        thisEntry = LogEntry(self, logDir)

        thisEntry["n_epoch"] = n_epoch
        thisEntry["n_iter"] = n_iter

        thisEntry.write()
        
        self.entries.append(thisEntry)

        self.lossBuffer = []

    def revert_state(self, argEntry):
        """
        revert the network to the state
        defined by a given log entry
        clear all following log entries        
        """
        for thisEntry in self.entries:
            if check_equality(thisEntry, argEntry):
                thisEntry.load()
        self.clear_after(argEntry)
        
    def clear(self):
        """
        Erase all saved checkpoints and update the network manifest
        """
        for thisEntry in self.entries:
            thisEntry.erase()

        self.entries = []
            
    def clear_after(self, argEntry):
        """
        Erase all saved checkpoints after a certain point
        and update the network manifest
        """

        startErasing = False
        toErase = []
        for thisEntry in self.entries:
            if startErasing:
                print ("erasing entry", thisEntry.outDir)
                toErase.append(thisEntry)
            if check_equality(thisEntry, argEntry):
                startErasing = True
        for thisEntry in toErase:
            self.entries.remove(thisEntry)
            thisEntry.erase()
        
    def get_loss(self):
        """
        return the loss time series
        """
        # there is a bug when the number of batches per checkpoint
        # is one.  This causes the entry loss series to have 1 dimension
        # instead of 2.
        loss_series = np.empty((0,3))
        for log_entry in self.entries:
            # if a loss file is missing for some reason, skip it
            try: 
                loss_series = np.concatenate([loss_series, log_entry.get_loss()])
            except ValueError:
                continue
        return loss_series
    
    def save_report(self):
        """
        Save the collated training data (train/eval loss)
        """
        loss_series = self.get_loss()
        np.savetxt(os.path.join(self.outDir, 'train_report.dat'),
                   loss_series)

class LogEntry:
    def __init__(self, manager, logDir):
        """
        A log entry contains information about the state
        of the network during training
        Each entry should contain enough information to 
        replicate the state totally, including RNG
        """
        self.manager = manager
        self.manifestPath = os.path.join(logDir, "logManifest.yaml")
        
        if os.path.exists(self.manifestPath):
            self.manifest = loadManifestDict(self.manifestPath)
        else:
            self.manifest = dict()

            self.manifest['numpyStatePath'] = os.path.join(logDir, "RNGstate.npy")
            self.manifest['torchStatePath'] = os.path.join(logDir, "RNGstate.torch")
            self.manifest['optimizerStatePath'] = os.path.join(logDir, "optimState.torch")
            self.manifest['checkpointPath'] = os.path.join(logDir, "weights.ckpt")
            self.manifest['loadOrderPath'] = os.path.join(logDir, "loadOrder.yaml")
            self.manifest['lossPath'] = os.path.join(logDir, "loss.dat")
        
        self.outDir = logDir

    def __getitem__(self, key):
        return self.manifest[key]

    def __setitem__(self, key, value):
        self.manifest[key] = value

    def write(self):
        """
        Write the state of the network/training proccess
        Include the current model weights, the numpy & torch RNG states,
        and a manifest describing it all
        """
        with open(self.manifestPath, 'w') as mf:
            print ('dumping checkpoint manifest to', self.manifestPath)
            yaml.dump(self.manifest, mf)

        self.manager.network.make_checkpoint(self.manifest['checkpointPath'])

        numpyState = np.random.get_state()
        np.save(self.manifest['numpyStatePath'], numpyState)

        torchState = torch.random.get_rng_state()
        torch.save(torchState, self.manifest['torchStatePath'])

        optimState = self.manager.network.optimizer.state_dict()
        torch.save(optimState, self.manifest['optimizerStatePath'])

        if self.manager.dataLoader:
            with open(self.manifest['loadOrderPath'], 'w') as lop:
                loadOrderDict = {"fileLoadOrder": self.manager.dataLoader.getFileLoadOrder(),
                                 "sampleLoadOrder": self.manager.dataLoader.getSampleLoadOrder(),
                                 }
                yaml.dump(loadOrderDict, lop)

        np.savetxt(self.manifest['lossPath'], self.manager.lossBuffer)

    def load(self):
        """
        Load this checkpoint and set all of the relevant
        parameters of the parent log manager and network
        TODO: when there are checkpoints chronologically after this one, delete them
        (should maybe handle loading from the log manager?)
        """

        self.manager.network.load_checkpoint(self.manifest['checkpointPath'])

        numpyState = np.load(self.manifest['numpyStatePath'],
                             allow_pickle = True)
        np.random.set_state(tuple(numpyState))

        torchState = torch.load(self.manifest['torchStatePath'])
        torch.random.set_rng_state(torchState)

        optimState = torch.load(self.manifest['optimizerStatePath'])
        self.manager.network.optimizer.load_state_dict(optimState)

        print ("loading to", self.manifest['n_epoch'], self.manifest['n_iter'])
        self.manager.network.n_epoch = self.manifest['n_epoch']
        self.manager.network.n_iter = self.manifest['n_iter']

        if self.manager.dataLoader:
            loadOrderDict = loadManifestDict(self.manifest['loadOrderPath'])
            self.manager.dataLoader.setFileLoadOrder(loadOrderDict['fileLoadOrder'])
            self.manager.dataLoader.setSampleLoadOrder(loadOrderDict['sampleLoadOrder'])

    def get_loss(self):
        return np.loadtxt(self.manifest['lossPath'])

    def erase(self):
        """
        Erase the log entry on disk
        """
        for leaf in os.listdir(self.outDir):
            os.remove(os.path.join(self.outDir, leaf))

        os.rmdir(self.outDir)

        return 
