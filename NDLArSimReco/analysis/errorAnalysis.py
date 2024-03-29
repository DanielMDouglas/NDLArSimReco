import numpy as np
import matplotlib.pyplot as plt
# from SLACplots.colors import *
# matplotlib.rc('text', **{"usetex": False})

import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import random

from NDLArSimReco.network import ConfigurableSparseNetwork
from NDLArSimReco.dataLoader import dataLoaderFactory
from NDLArSimReco.utils import sparseTensor

import yaml
import os

# from SLACplots.colors import *

import MinkowskiEngine as ME
ME.set_sparse_tensor_operation_mode(ME.SparseTensorOperationMode.SHARE_COORDINATE_MANAGER)

def main(args):
    with open(args.manifest) as mf:
        manifest = yaml.load(mf, Loader = yaml.FullLoader)

    print ("initializing network...")
    net = ConfigurableSparseNetwork(D=3, manifest = manifest, make_output = False).to(device)
    
    if args.checkpoint:
        net.load_checkpoint(args.checkpoint)
    
    infilePath = manifest['testfilePath'] 

    if os.path.isdir(infilePath[0]):
        infileList = [os.path.join(infilePath[0], thisFile) 
                      for thisFile in os.listdir(infilePath[0])]
        print ("loading files from directory", infileList)
    else:
        infileList = infilePath
        print ("loading files from list", infileList)
    dl = dataLoaderFactory[manifest['dataLoader']](infileList,
                                                   batchSize = manifest['batchSize'])
    # dl = dataLoaderFactory[manifest['dataLoader']](infileList,
    #                                                batchSize = 16)

    dl.genFileLoadOrder()

    transform = sparseTensor.transformFactory[manifest['transform']]()
    hits, edep = next(dl.load(transform = transform))
    
    net.eval()
    prediction = net(hits)
    predMean, predStd, predOcc = net.criterion.feature_map(prediction)
    predMask = predOcc > 0.5
    predMean = torch.where(predMask, predMean, torch.zeros_like(predMean))
    # predStd = torch.where(predMask, predStd, torch.ones_like(predStd))
    predStd = torch.where(predMask, predStd, 0.1*torch.ones_like(predStd))
    # predStd /= (4*predMean + 2)
    print ('loss', net.criterion(prediction, edep))

    # Ereco = predMean.detach().numpy()
    # Ereco = predMean.detach().numpy() + 0.773*predStd.detach().numpy()
    Ereco = predMean.detach().numpy()
    error = predStd.detach().numpy()

    Etrue = edep.features.detach().numpy()[:,0]
    Etrue = np.max([Etrue, np.zeros_like(Etrue)], axis = 0)

    # nDropoutPredictions = 10

    # MCDmeanPred = []
    # MCDstdPred = []

    # net.MCdropout()
    # for _ in range(nDropoutPredictions):
    #     thisPred = net(hits)
    #     thisPredMean, thisPredStd = net.criterion.feature_map(thisPred)

    #     MCDmeanPred.append(thisPredMean.detach().numpy())
    #     MCDstdPred.append(thisPredStd.detach().numpy())

    # MCDmeanPredictions = np.mean(MCDmeanPred)
    # MCDstdPredictions = np.sqrt(sum(STDi**2 for STDi in MCDstdPred))

    scatFig = plt.figure()
    scatAx = scatFig.gca()
    # scatAx.scatter(Ereco, error)
    # scatAx.scatter(Etrue, error)
    scatAx.hist2d(np.abs(Etrue - Ereco), error)
    
    scatFig.savefig('figs/reco_vs_error.png')
    
    Efig = plt.figure()
    Eax = Efig.gca()

    Errfig = plt.figure()
    ErrAx = Errfig.gca()

    Pullfig = plt.figure()
    PullAx = Pullfig.gca()
    # Ebins = np.linspace(0, 2, 30)
    # Ebins = np.linspace(0, 5, 51)
    Ebins = np.linspace(0.2, 1.8, 49)
    # Ebins = np.linspace(0.2, 1.8, 26)
    
    # epsilon = 1.e-2
    Eax.hist(Etrue, histtype = 'step', label = 'Ground Truth', bins = Ebins)

    # Ereco = prediction.features.detach().numpy()[:,0]
    # Ereco = np.max([Ereco, np.zeros_like(Ereco)], axis = 0)
    # epsilon = 1.e-2
    # error = np.exp(prediction.features.detach().numpy()[:,1]) + epsilon
    # error = torch.relu(prediction.features[:,1]).detach().numpy() + epsilon

    # resid = (edep - prediction).features.detach().numpy()[:,0]
    
    Eax.hist(Ereco,
             histtype = 'step',
             label = 'Prediction',
             bins = Ebins)
    # Eax.hist(meanPredictions,
    #          histtype = 'step',
    #          label = 'Dropout Prediction Mean',
    #          bins = Ebins)

    Efig.legend()
    Eax.set_xlabel(r'Energy per Voxel [MeV]')
    Efig.savefig("figs/Ehist.png")

    # ErrBins = np.linspace(0, 1.25, 30)
    
    # counts, _, _ = ErrAx.hist(error,
    #            histtype = 'step',
    #            label = 'Predicted Uncertainty',
    #            bins = ErrBins)
    # print (sum(counts))
    # counts, _, _ = ErrAx.hist(np.abs(resid),
    #            histtype = 'step',
    #            label = 'G.T. Error (|prediction residual|)',
    #            bins = ErrBins)
    # print (sum(counts))
    # counts, _, _ = ErrAx.hist(stdPredictions,
    #            histtype = 'step',
    #            label = 'Model Uncertainty Estimate (50 predictions with dropout)',
    #            bins = ErrBins)
    # print (sum(counts))

    # combinedError = np.sqrt(np.power(error, 2) + np.power(stdPredictions, 2))
    # counts, _, _ = ErrAx.hist(combinedError,
    #                           histtype = 'step',
    #                           label = 'Combined Uncertainty (quad. sum)',
    #                           bins = ErrBins)
    # print (sum(counts))

    # ErrAx.set_xlabel(r'Error per Voxel [MeV]')
    # Errfig.legend()
    # Errfig.savefig("figs/Error.png")


    # # Etrue = edep.features[np.lexsort(edep.coordinates.numpy().T)].detach().numpy()[:,0]
    # # Ereco = prediction.features[np.lexsort(prediction.coordinates.numpy().T)].detach().numpy()[:,0]
    # coords = edep.coordinates
    # Etrue = edep.features_at_coordinates(coords.float())[:,0].detach().numpy()
    # Ereco = prediction.features_at_coordinates(coords.float())[:,0].detach().numpy()
    # # error = prediction.features_at_coordinates(coords.float())[:,1].detach().numpy()
    # # error = np.exp(error) + epsilon
    # error = torch.relu(prediction.features_at_coordinates(coords.float())[:,1]).detach().numpy() + epsilon
    resid = Etrue[predMask] - Ereco[predMask]
    # print (Etrue.shape)
    # print (Ereco.shape)

    PullAx.hist(resid/error[predMask],
                density = True,
                histtype = 'step',
                label = 'residual / predicted uncertainty',
                bins = np.linspace(-3, 3, 50),
                )
    # PullAx.hist(resid/stdPredictions,
    #             density = True,
    #             histtype = 'step',
    #             label = 'residual / model uncertainty',
    #             bins = np.linspace(-3, 3, 50),
    #             )
    import scipy.stats as st
    zSpace = np.linspace(-3, 3, 1000)
    PullAx.plot(zSpace, st.norm.pdf(zSpace))
    PullAx.set_xlabel(r'Prediction Z-Score')
    Pullfig.legend()
    Pullfig.savefig("figs/pull.png")

    meanCorrFig = plt.figure()
    meanCorrAx = meanCorrFig.gca()
    from matplotlib import colors

    from sklearn.linear_model import LinearRegression
    # reg = LinearRegression().fit(np.array([Etrue, Ereco]).T)
    # reg = LinearRegression(fit_intercept = False).fit(Etrue.reshape(-1, 1), Ereco,
    #                              )
    reg = LinearRegression(fit_intercept = True).fit(Etrue.reshape(-1, 1), Ereco)
    # print ("fitted intercept", reg.intercept_)
    # print ("fitted slope", reg.coef_)
    
    meanSpace = np.linspace(0, Ebins[-1], 1000)

    # mask = np.logical_or(Etrue > 0.25, Ereco > 0.25)
    # Etrue = Etrue[mask]
    # Ereco = Ereco[mask]
    
    result = meanCorrAx.hist2d(Etrue, Ereco,
                               bins = (Ebins, Ebins),
                               norm = colors.LogNorm(),
                               )
    # print ("result", result)
    meanCorrAx.plot(meanSpace, meanSpace, ls = '--', c = 'red')
    # meanCorrAx.plot(meanSpace, meanSpace*reg.coef_ + reg.intercept_, ls = '--', c = 'orange')
    meanCorrAx.plot(meanSpace, reg.intercept_ + meanSpace*reg.coef_, ls = '--', c = 'orange')
    meanCorrAx.set_xlabel(r'G.T. Energy per Voxel [MeV]')
    meanCorrAx.set_ylabel(r'Predicted Energy per Voxel [MeV]')
    plt.colorbar(result[-1])
    meanCorrFig.savefig("figs/meanCorr.png")

    meanTrendFig = plt.figure()
    meanTrendAx = meanTrendFig.gca()
    from scipy.stats import binned_statistic
    bin_means, bin_edges, binnumber = binned_statistic(Etrue, Ereco,
                                                       statistic = 'mean',
                                                       bins = Ebins)
    bin_medians, bin_edges, binnumber = binned_statistic(Etrue, Ereco,
                                                       statistic = 'median',
                                                       bins = Ebins)
    bin_std, bin_edges, binnumber = binned_statistic(Etrue, Ereco,
                                                     statistic = 'std',
                                                     bins = Ebins)
    bin_lowCI, bin_edges, binnumber = binned_statistic(Etrue, Ereco,
                                                       statistic = lambda x: np.quantile(x, 0.16),
                                                       bins = Ebins)
    bin_highCI, bin_edges, binnumber = binned_statistic(Etrue, Ereco,
                                                        statistic = lambda x: np.quantile(x, 0.84),
                                                        bins = Ebins)
    
    print (bin_means, bin_edges, binnumber)
    bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])

    # print (bin_centers.shape, bin_means.shape)
    print (bin_centers)
    meanTrendAx.plot(meanSpace, meanSpace, ls = '--', c = 'red')
    # meanTrendAx.errorbar(bin_centers, bin_means,
    #                      yerr = bin_std,
    #                      fmt = 'o')
    meanTrendAx.errorbar(bin_centers, bin_means,
                         yerr = (bin_medians - bin_lowCI, bin_highCI - bin_medians),
                         fmt = 'o')
    meanTrendAx.set_xlabel(r'G.T. Energy per Voxel [MeV]')
    meanTrendAx.set_ylabel(r'Predicted Energy per Voxel [MeV]')

    # meanTrendAx.scatter(bin_centers, bin_centers)
    meanTrendFig.savefig("figs/meanTrend.png")
    
    import uncertainty_toolbox as uct
    # # Ereco = prediction.features.detach()[:,0]
    # # Ereco = torch.ReLU(Ereco)
    # # epsilon = 1.e-2
    # # error = torch.exp(prediction.features.detach()[:,1]) + epsilon
    # # Etrue = edep.features.detach()[:,0]
    # # Etrue = torch.ReLU(Etrue)
    # # uct.viz.plot_calibration(Ereco, error, Etrue-Ereco)
    uct.viz.plot_calibration(Ereco[predMask], error[predMask], Etrue[predMask])
    plt.savefig("figs/calibration.png")

    uct.viz.plot_intervals(Ereco, error, Etrue, n_subset = 50)
    plt.savefig("figs/intervals.png")

    errBins = np.linspace(0, 2, 51)
    plt.figure()
    plt.hist2d(np.abs(Ereco[predMask] - Etrue[predMask]), error[predMask],
               norm = colors.LogNorm(),
               bins = (errBins, errBins))
    plt.plot(np.linspace(0, 2, 1000),
             np.linspace(0, 2, 1000),
             ls = '--',
             color = 'red')
    plt.xlabel(r'True Error $|\hat{E} - E|$ [MeV]')
    plt.ylabel(r'Predicted Error $\sigma$ [MeV]')
    plt.savefig("figs/errDist.png")
    
    uct.viz.plot_intervals_ordered(Ereco[predMask], error[predMask], Etrue[predMask], n_subset = 50)
    # uct.viz.plot_intervals_ordered(Ereco, error, Etrue)
    plt.savefig("figs/intervals_ordered.png")

    # # pnn_metrics = uct.metrics.get_all_metrics(Ereco, error, Etrue-Ereco)
    pnn_metrics = uct.metrics.get_all_metrics(Ereco[predMask], error[predMask], Etrue[predMask])

    # # uct.viz.plot_calibration(Ereco, error, Etrue)
    # # plt.savefig("figs/calibration.png")
    
    # plt.show()
    # return 

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--manifest', type = str,
                        default = "/sdf/home/d/dougl215/studies/NDLArSimReco/NDLArSimReco/manifests/testManifest.yaml",
                        help = "network manifest yaml file")
    parser.add_argument('-c', '--checkpoint', type = str,
                        default = "/sdf/home/d/dougl215/studies/NDLArSimReco/NDLArSimReco/checkpoints/checkpoint_final_10_0.ckpt",
                        help = "checkpoint file to use")
    
    args = parser.parse_args()

    main(args)
