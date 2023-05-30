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
from NDLArSimReco.dataLoader import DataLoader

import yaml
import os

import MinkowskiEngine as ME
ME.set_sparse_tensor_operation_mode(ME.SparseTensorOperationMode.SHARE_COORDINATE_MANAGER)

def main(args):
    with open(args.manifest) as mf:
        manifest = yaml.load(mf, Loader = yaml.FullLoader)

    print ("initializing network...")
    net = ConfigurableSparseNetwork(in_feat=1, D=3, manifest = manifest, make_output = False).to(device)
    
    if args.checkpoint:
        net.load_checkpoint(args.checkpoint)
    
    infileList = manifest['testfilePath']
    dl = DataLoader(infileList, batchSize = 50)

    dl.setFileLoadOrder()

    hits, edep = next(dl.load())

    net.eval()
    prediction = net(hits)
    predMean, predStd = net.criterion.feature_map(prediction)

    # Ereco = predMean.detach().numpy()
    Ereco = predMean.detach().numpy() + 0.773*predStd.detach().numpy()
    
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
    
    Efig = plt.figure()
    Eax = Efig.gca()

    Errfig = plt.figure()
    ErrAx = Errfig.gca()

    Pullfig = plt.figure()
    PullAx = Pullfig.gca()
    # Ebins = np.linspace(0, 2, 30)
    Ebins = np.linspace(0, 5, 50)
    
    Etrue = edep.features.detach().numpy()[:,0]
    Etrue = np.max([Etrue, np.zeros_like(Etrue)], axis = 0)
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
    # resid = Etrue - Ereco
    # print (Etrue.shape)
    # print (Ereco.shape)

    # PullAx.hist(resid/error,
    #             density = True,
    #             histtype = 'step',
    #             label = 'residual / predicted uncertainty',
    #             bins = np.linspace(-3, 3, 50),
    #             )
    # PullAx.hist(resid/stdPredictions,
    #             density = True,
    #             histtype = 'step',
    #             label = 'residual / model uncertainty',
    #             bins = np.linspace(-3, 3, 50),
    #             )
    # import scipy.stats as st
    # zSpace = np.linspace(-3, 3, 1000)
    # PullAx.plot(zSpace, st.norm.pdf(zSpace))
    # PullAx.set_xlabel(r'Prediction Z-Score')
    # Pullfig.legend()
    # Pullfig.savefig("figs/pull.png")

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
    
    # import uncertainty_toolbox as uct
    # # Ereco = prediction.features.detach()[:,0]
    # # Ereco = torch.ReLU(Ereco)
    # # epsilon = 1.e-2
    # # error = torch.exp(prediction.features.detach()[:,1]) + epsilon
    # # Etrue = edep.features.detach()[:,0]
    # # Etrue = torch.ReLU(Etrue)
    # # uct.viz.plot_calibration(Ereco, error, Etrue-Ereco)
    # uct.viz.plot_calibration(Ereco, error, Etrue)
    # plt.savefig("figs/calibration.png")

    # uct.viz.plot_intervals(Ereco, error, Etrue, n_subset = 20)
    # plt.savefig("figs/intervals.png")

    # uct.viz.plot_intervals_ordered(Ereco, error, Etrue, n_subset = 20)
    # # uct.viz.plot_intervals_ordered(Ereco, error, Etrue)
    # plt.savefig("figs/intervals_ordered.png")

    # # pnn_metrics = uct.metrics.get_all_metrics(Ereco, error, Etrue-Ereco)
    # pnn_metrics = uct.metrics.get_all_metrics(Ereco, error, Etrue)

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
