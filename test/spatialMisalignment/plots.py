import numpy as np
import matplotlib.pyplot as plt

def plot_deflection_map(edep_position, difference, plotField = 'x'):
    zSliceInd = 25
    if plotField == 'mag':
        field = np.sqrt(np.sum(np.power(difference, 2), axis = -1))[:,:,zSliceInd].T
        vmax = np.max(field[~np.isnan(field)])
        vmin = 0
        cmap = 'viridis'
        cbLabel = '|shift| [mm]'
    elif plotField == 'x':
        field = difference[:, :, zSliceInd, 0].T
    
        vmax = max(np.max(field[~np.isnan(field)]),
               abs(np.min(field[~np.isnan(field)])))
        vmin = -vmax
        cmap = 'bwr'
        cbLabel = r'$\Delta x$ [mm]'
    elif plotField == 'y':
        field = difference[:, :, zSliceInd, 1].T
    
        vmax = max(np.max(field[~np.isnan(field)]),
               abs(np.min(field[~np.isnan(field)])))
        vmin = -vmax
        cmap = 'bwr'
        cbLabel = r'$\Delta y$ [mm]'
    elif plotField == 'z':
        field = difference[:, :, zSliceInd, 2].T
    
        vmax = max(np.max(field[~np.isnan(field)]),
               abs(np.min(field[~np.isnan(field)])))
        vmin = -vmax
        cmap = 'bwr'
        cbLabel = r'$\Delta z$ [mm]'

    xMin = np.min(edep_position[:, :, :, 0])
    xMax = np.max(edep_position[:, :, :, 0])

    yMin = np.min(edep_position[:, :, :, 1])
    yMax = np.max(edep_position[:, :, :, 1])

    zMin = np.min(edep_position[:, :, :, 2])
    zMax = np.max(edep_position[:, :, :, 2])

    fig = plt.figure()
    ax = fig.gca()
    
    im = ax.imshow(field,
                   origin = 'lower',
                   vmin = vmin,
                   vmax = vmax,
                   cmap = cmap,
                   extent = (xMin, xMax, yMin, yMax))
    
    ax.set_xlabel(r'x [mm]')
    ax.set_ylabel(r'y [mm]')
    
    cb = plt.colorbar(im)
    cb.set_label(cbLabel)

    plt.tight_layout()

    return ax

def plot_deflection_distribution(difference, plotField = 'x'):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    if plotField == 'mag':
        field = np.sqrt(np.sum(np.power(difference, 2),
                               axis = -1)).flatten()
    elif plotField == 'x':
        field = difference[:,:,:,0].flatten()
    elif plotField == 'y':
        field = difference[:,:,:,1].flatten()
    elif plotField == 'z':
        field = difference[:,:,:,2].flatten()

    print (field)
    ax.hist(field,
            histtype = 'step',
            bins = np.logspace(-1, 2, 20))
    # ax.semilogy()
    ax.loglog()

    ax.set_xlabel(r'Displacement Magnitude [mm]')
    
    plt.tight_layout()

    return ax

def main(args):
    edep_position = np.load(args.edep)
    hits_position = np.load(args.hits)

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection = '3d')

    # ax.quiver(*edep_position.T, *difference.T)

    edep_position = edep_position.reshape((70, 30, 50, 3))
    hits_position = hits_position.reshape((70, 30, 50, 3))

    difference = hits_position - edep_position

    plot_deflection_map(edep_position, difference)
    plot_deflection_map(edep_position, difference, plotField = 'mag')
    plot_deflection_distribution(difference, plotField = 'mag')
    
    plt.show()
    

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--edep', type = str,
                        help = "edep positions")
    parser.add_argument('--hits', type = str,
                        help = "hit positions (indexed parallel to edep)")
    args = parser.parse_args()

    main(args)
