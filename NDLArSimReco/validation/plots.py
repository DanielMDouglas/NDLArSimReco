import numpy as np

import yaml

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# from SLACplots import colors

import h5py

from particle import Particle    

class ValidationPlot:
    def __init__(self):
        self.fig = plt.figure()
        
        self.data = []
        
    def update(self, dataFile):
        self.data.append(datafile)

    def draw(self):
        pass

class SpatialDistributionPlot (ValidationPlot):
    def __init__(self):
        super().__init__()
        
        gs = GridSpec(1, 3,
                      figure = self.fig,
                      hspace = 0)
        self.axX = self.fig.add_subplot(gs[:,0])
        self.axY = self.fig.add_subplot(gs[:,1],
                                        sharey = self.axX)
        self.axZ = self.fig.add_subplot(gs[:,2],
                                        sharey = self.axX)

        plt.setp(self.axY.get_yticklabels(),
                 visible=False)
        plt.setp(self.axZ.get_yticklabels(),
                 visible=False)

        self.fig.suptitle('Track Endpoint Positions',
                          size=20)

        self.xBins = np.linspace(-290, 390, 10)
        self.yBins = np.linspace(-210, 70, 10)
        self.zBins = np.linspace(425, 905, 10)

        self.axX.set_xlabel(r'x [mm]')
        self.axY.set_xlabel(r'y [mm]')
        self.axZ.set_xlabel(r'z [mm]')

        self.x = []
        self.y = []
        self.z = []
        
    def update(self, dataFile):
        traj = dataFile['trajectories']

        primaries = traj[traj['parentID'] == -1]

        x, y, z = primaries['xyz_start'].T/10

        self.x.append(x)
        self.y.append(y)
        self.z.append(z)

    def draw(self):
        self.axX.hist(self.x,
                      histtype = 'step',
                      bins = self.xBins)
    
        self.axY.hist(self.y,
                      histtype = 'step',
                      bins = self.yBins)
    
        self.axZ.hist(self.z,
                      histtype = 'step',
                      bins = self.zBins)

        plt.tight_layout()

class MultiplicityPlot (ValidationPlot):
    def __init__(self):
        super().__init__()
        
        self.ax = self.fig.gca()

        self.ax.set_title(r'N Particles per Event')
        self.ax.set_xlabel(r'N Particles')

        self.nPart = []
        
    def update(self, dataFile):
        traj = dataFile['trajectories']

        for evID in np.unique(traj['eventID']):
            evTraj = traj[traj['eventID'] == evID]
            self.nPart.append(len(np.unique(evTraj['trackID'])))

    def draw(self):
        self.ax.hist(self.nPart)

        plt.tight_layout()

class PrimaryTypePlot (ValidationPlot):
    def __init__(self):
        super().__init__()
        
        self.ax = self.fig.gca()

        self.ax.set_title(r'Primary Particle Type')

        self.primaryPDG = []

    def update(self, dataFile):
        traj = dataFile['trajectories']

        primaries = traj[traj['parentID'] == -1]
        primaryPDG = primaries['pdgId']

        self.primaryPDG += list(primaryPDG)

    def draw(self):
        unqPDG = np.unique(self.primaryPDG)
        pdgInd = [list(unqPDG).index(p) for p in self.primaryPDG]

        pdgBins = np.linspace(np.min(pdgInd),
                              np.max(pdgInd)+1,
                              len(unqPDG)+1)
        self.ax.hist(pdgInd,
                     bins = pdgBins,
                     histtype = 'barstacked',
                     rwidth = 0.8)
        
        LABELS = [r'$'+Particle.from_pdgid(up).latex_name+r'$'
                  for up in unqPDG]
        self.ax.set_xticks(np.arange(len(unqPDG))+0.5)
        self.ax.set_xticklabels(LABELS)

        plt.tight_layout()

class MomentumPrimary2dPlot (ValidationPlot):
    def __init__(self):
        super().__init__()
        
        self.ax = self.fig.gca()

        self.primaryPDG = []
        self.pmag = []

    def update(self, dataFile):
        traj = dataFile['trajectories']

        primaries = traj[traj['parentID'] == -1]
        primaryPDG = primaries['pdgId']

        self.primaryPDG += list(primaryPDG)        
        self.pmag += list(np.linalg.norm(primaries['pxyz_start'], axis = 1))

    def draw(self):
        unqPDG = np.unique(self.primaryPDG)
        pdgInd = [list(unqPDG).index(p) for p in self.primaryPDG]

        pdgBins = np.linspace(np.min(pdgInd),
                              np.max(pdgInd)+1,
                              len(unqPDG)+1)
        
        pbins = np.linspace(0, np.max(self.pmag), 15)

        counts, xbins, ybins, mesh = self.ax.hist2d(pdgInd, self.pmag,
                                             bins = (pdgBins, pbins))

        self.fig.colorbar(mesh)

        self.ax.set_xlabel(r'Primary Type')
        self.ax.set_ylabel(r'Primary $|p|$ [MeV/c]')

        LABELS = [r'$'+Particle.from_pdgid(up).latex_name+r'$'
                  for up in unqPDG]
        self.ax.set_xticks(np.arange(len(unqPDG))+0.5)
        self.ax.set_xticklabels(LABELS)

        plt.tight_layout()

class MomentumPrimary1dPlot (ValidationPlot):
    def __init__(self):
        super().__init__()
        
        self.ax = self.fig.gca()

        self.pmag = {}

    def update(self, dataFile):
        traj = dataFile['trajectories']

        primaries = traj[traj['parentID'] == -1]
        primaryPDG = primaries['pdgId']

        for thisPrimary in np.unique(primaryPDG):
            if not thisPrimary in self.pmag:
                self.pmag.update({thisPrimary: []})

            primariesOfType = primaries[primaryPDG == thisPrimary]
            self.pmag[thisPrimary] += list(np.linalg.norm(primariesOfType['pxyz_start'],
                                                          axis = 1))
        
    def draw(self):
        pdgInd = np.arange(len(self.pmag.keys()))

        pbins = np.linspace(0, np.max(np.concatenate(list(self.pmag.values()))), 15)

        self.ax.hist(np.concatenate(list(self.pmag.values())),
                     bins = pbins,
                     label = r'Total',
                     histtype = 'step',
                     color = colors.SLACgrey,
                     lw = 3)
        for key, value in self.pmag.items():
            self.ax.hist(value, bins = pbins,
                         label = r'$'+Particle.from_pdgid(key).latex_name+r'$',
                         histtype = 'step',
                         lw = 3)

        self.ax.set_xlabel(r'Primary $|p|$ [MeV/c]')
        self.ax.legend()

        plt.tight_layout()

class EnergyPrimary2dPlot (ValidationPlot):
    def __init__(self):
        super().__init__()
        
        self.ax = self.fig.gca()

        self.primaryPDG = []
        self.Ek = []

    def update(self, dataFile):
        traj = dataFile['trajectories']

        primaries = traj[traj['parentID'] == -1]
        primaryPDG = primaries['pdgId']

        self.primaryPDG += list(primaryPDG)        

        pmag = np.linalg.norm(primaries['pxyz_start'], axis = 1)
        m = np.array([Particle.from_pdgid(p).mass
                      for p in primaryPDG])
        Ek = np.sqrt(np.power(pmag, 2) + np.power(m, 2)) - m
        self.Ek += list(Ek)

    def draw(self):
        unqPDG = np.unique(self.primaryPDG)
        pdgInd = [list(unqPDG).index(p) for p in self.primaryPDG]

        pdgBins = np.linspace(np.min(pdgInd),
                              np.max(pdgInd)+1,
                              len(unqPDG)+1)
        
        Ebins = np.linspace(0, np.max(self.Ek), 15)

        counts, xbins, ybins, mesh = self.ax.hist2d(pdgInd, self.Ek,
                                             bins = (pdgBins, Ebins))

        self.fig.colorbar(mesh)

        self.ax.set_xlabel(r'Primary Type')
        self.ax.set_ylabel(r'Primary $E_k$ [MeV]')
        
        LABELS = [r'$'+Particle.from_pdgid(up).latex_name+r'$'
                  for up in unqPDG]
        self.ax.set_xticks(np.arange(len(unqPDG))+0.5)
        self.ax.set_xticklabels(LABELS)

        plt.tight_layout()

class EnergyPrimary1dPlot (ValidationPlot):
    def __init__(self):
        super().__init__()
        
        self.ax = self.fig.gca()

        self.Ek = {}

    def update(self, dataFile):
        traj = dataFile['trajectories']

        primaries = traj[traj['parentID'] == -1]
        primaryPDG = primaries['pdgId']

        for thisPrimary in np.unique(primaryPDG):
            if not thisPrimary in self.Ek:
                self.Ek.update({thisPrimary: []})

            primariesOfType = primaries[primaryPDG == thisPrimary]
            pmag = np.linalg.norm(primariesOfType['pxyz_start'],
                                  axis = 1)
            m = Particle.from_pdgid(thisPrimary).mass
            Ek = np.sqrt(np.power(pmag, 2) + np.power(m, 2)) - m
            self.Ek[thisPrimary] += list(Ek)

            
    def draw(self):
        pdgInd = np.arange(len(self.Ek.keys()))

        Ebins = np.linspace(0, np.max(np.concatenate(list(self.Ek.values()))), 15)

        self.ax.hist(np.concatenate(list(self.Ek.values())),
                     bins = Ebins,
                     label = r'Total',
                     histtype = 'step',
                     color = colors.SLACgrey,
                     lw = 3)
        for key, value in self.Ek.items():
            self.ax.hist(value, bins = Ebins,
                         label = r'$'+Particle.from_pdgid(key).latex_name+r'$',
                         histtype = 'step',
                         lw = 3)

        self.ax.set_xlabel(r'Primary $E_k$ [MeV]')
        self.ax.legend()

        plt.tight_layout()

class EdepRatioPrimary1dPlot (ValidationPlot):
    def __init__(self):
        super().__init__()
        
        self.ax = self.fig.gca()

        self.EdepRatio = {}

    def update(self, dataFile):
        tracks = dataFile['tracks']
        traj = dataFile['trajectories']

        primaries = traj[traj['parentID'] == -1]
        
        primaryPDG = primaries['pdgId']
        
        primPmag = np.linalg.norm(primaries['pxyz_start'], axis = 1)
        primMass = np.array([Particle.from_pdgid(p['pdgId']).mass for p in primaries])
        Einit = np.sqrt(np.power(primPmag, 2) + np.power(primMass, 2))

        Edep = []
        for eid in np.unique(traj['eventID']):
            evTracks = tracks[tracks['eventID'] == eid]
            if len(evTracks):
                totEdep = np.sum(evTracks['dE'])
            else:
                totEdep = 0
            Edep.append(totEdep)
        Edep = np.array(Edep)
        ratio = Edep/Einit
        
        for thisPrimary in np.unique(primaryPDG):
            if not thisPrimary in self.EdepRatio:
                self.EdepRatio.update({thisPrimary: []})

            thisPrimRatio = ratio[primaryPDG == thisPrimary]
            thisPrimRatio = thisPrimRatio[thisPrimRatio > 0]
            self.EdepRatio[thisPrimary] += list(thisPrimRatio)
            
    def draw(self):
        pdgInd = np.arange(len(self.EdepRatio.keys()))

        ratioBins = np.linspace(0, np.max(np.concatenate(list(self.EdepRatio.values()))), 11)

        self.ax.hist(np.concatenate(list(self.EdepRatio.values())),
                     bins = ratioBins,
                     label = r'Total',
                     histtype = 'step',
                     color = colors.SLACgrey,
                     lw = 3)
        for key, value in self.EdepRatio.items():
            self.ax.hist(value, bins = ratioBins,
                         label = r'$'+Particle.from_pdgid(key).latex_name+r'$',
                         histtype = 'step',
                         lw = 3)

        self.ax.semilogy()
            
        self.ax.set_xlabel(r'$E_{\mathrm{dep.}}/E_{\mathrm{init.}}$')
        self.ax.legend(loc = 'lower right')

        plt.tight_layout()

class EdepTrackLength2dPlot (ValidationPlot):
    def __init__(self):
        super().__init__()
        
        self.ax = self.fig.gca()

        self.trackLen = []
        self.trackdE = []

    def update(self, dataFile):
        tracks = dataFile['tracks']

        for eid in np.unique(tracks['eventID']):
            evTracks = tracks[tracks['eventID'] == eid]
            for tid in np.unique(evTracks['trackID']):
                trackSegs = evTracks[evTracks['trackID'] == tid]
        
                self.trackLen.append(np.sum(trackSegs['dx']))
                self.trackdE.append(np.sum(trackSegs['dE']))

    def draw(self):
        counts, xbins, ybins, mesh = self.ax.hist2d(self.trackLen, self.trackdE,
                                                    bins = (np.logspace(-1, np.log10(np.max(self.trackLen)), 15),
                                                            np.logspace(-1, np.log10(np.max(self.trackdE)), 15),
                                                            ),
                                                    norm = mpl.colors.LogNorm())
        self.fig.colorbar(mesh)
        self.ax.loglog()
        self.ax.set_xlabel(r'Track Length [mm]')
        self.ax.set_ylabel(r'Track dE [MeV]')

        plt.tight_layout()

def makePlot(plotClass, infileList):
    plotInstance = plotClass()
    for infile in infileList:
        plotInstance.update(infile)
    plotInstance.draw()
    
def main(args):
    # make output structure

    # load the file(s)
    with open(args.manifest) as mf:
        manifest = yaml.load(mf, Loader = yaml.FullLoader)

    f = h5py.File(manifest['trainfile'])

    plots = [SpatialDistributionPlot(),
             MultiplicityPlot(),
             PrimaryTypePlot(),
             MomentumPrimary2dPlot(),
             MomentumPrimary1dPlot(),
             EnergyPrimary2dPlot(),
             EnergyPrimary1dPlot(),
             EdepRatioPrimary1dPlot(),
             EdepTrackLength2dPlot(),
             ]

    for thisPlot in plots:
        thisPlot.update(f)
        thisPlot.draw()    
    
    # fig = plt.figure()

    # ratioBins = np.linspace(0, 1, 11)

    # Edep = []
    # Einit = []
    # ratio = []
    # for eid in np.unique(tracks['eventID']):
    #     evTracks = tracks[tracks['eventID'] == eid]
    #     totEdep = np.sum(evTracks['dE'])
    #     Edep.append(totEdep)

    #     evTraj = traj[traj['eventID'] == eid]
    #     evPrim = evTraj[evTraj['parentID'] == -1]
        
    #     primPmag = np.linalg.norm(evPrim['pxyz_start'], axis = 1)
    #     primMass = Particle.from_pdgid(evPrim['pdgId']).mass
    #     evEinit = (np.sqrt(np.power(primPmag, 2) + np.power(primMass, 2)))[0]
    #     Einit.append(evEinit)

    #     ratio.append(totEdep/evEinit)

    # plt.hist(ratio,
    #          histtype = 'step',
    #          label = 'Total',
    #          bins = ratioBins,
    #          color = colors.SLACgrey,
    #          lw = 3)

    # Edep = []
    # Einit = []
    # ratio = []
    # for thisPDG in unqPDG:
    #     for eid in np.unique(tracks['eventID']):
    #         evTracks = tracks[tracks['eventID'] == eid]
    #         totEdep = np.sum(evTracks['dE'])
    #         Edep.append(totEdep)
            
    #         evTraj = traj[traj['eventID'] == eid]
    #         evPrim = evTraj[evTraj['parentID'] == -1]
            
    #         primPmag = np.linalg.norm(evPrim['pxyz_start'], axis = 1)
    #         primMass = Particle.from_pdgid(evPrim['pdgId']).mass
    #         evEinit = (np.sqrt(np.power(primPmag, 2) + np.power(primMass, 2)))[0]
    #         Einit.append(evEinit)

    #         if evPrim['pdgId'] == thisPDG:
    #             ratio.append(totEdep/evEinit)

    #     plt.hist(ratio,
    #              histtype = 'step',
    #              label = r'$'+Particle.from_pdgid(thisPDG).latex_name+r'$',
    #              bins = ratioBins,
    #              lw = 3)

    # plt.semilogy()
    # plt.xlabel(r'$E_{\mathrm{dep.}}/E_{\mathrm{init.}}$')
    # plt.legend(loc = 'lower right')
    # plt.tight_layout()
            
    plt.show()
    
if __name__ == '__main__':
    
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--manifest', type = str,
                        default = "/home/dan/studies/NDLArForwardME/testManifest.yaml",
                        help = "network manifest yaml file")
    parser.add_argument('-v', '--verbose',
                        action = 'store_true',
                        help = "print extra debug messages")
    
    
    args = parser.parse_args()
    
    main(args)

    
