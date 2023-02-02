import torch
import torch.nn as nn

# criterion = nn.CrossEntropyLoss()
# criterion = nn.MSELoss()
# criterion = lambda x, y: 0

# def normalP(z, s):
#     return torch.exp(-torch.pow(z/s, 2)/2.)/(s*np.sqrt(2*torch.pi))

# def LLH(output, truth):
#     # print ("prediction", np.sum(prediction.coordinates[:,1:]))

#     # smear prediction with gaussian to do a likelihood style
#     gaussWidth = 5
#     jointLogLikelihood = 0
#     totalEMD = 0
#     totalEtrue = []
#     totalEpred = []
#     for batchNo in torch.unique(output.coordinates[:,0]):
#         batchPredCoords = output.coordinates[output.coordinates[:,0] == batchNo][:,1:].float()
#         batchTrueCoords = truth.coordinates[truth.coordinates[:,0] == batchNo][:,1:].float()

#         nTrue = batchTrueCoords.shape[0]
#         nPred = batchPredCoords.shape[0]

#         batchPredE = output.features[output.coordinates[:,0] == batchNo][:,0]
#         # batchPredProb = output.features[output.coordinates[:,0] == batchNo][:,1]
#         batchTrueE = truth.features[truth.coordinates[:,0] == batchNo]

#         batchPredEtotal = torch.sum(batchPredE)
#         batchTrueEtotal = torch.sum(batchTrueE)

#         normedBatchPredE = torch.abs(batchPredE/batchPredEtotal)
#         normedBatchTrueE = batchTrueE/batchTrueEtotal

#         # mags = torch.prod(torch.stack((normedBatchPredE.repeat((nTrue, 1)),
#         #                                torch.swapaxes(normedBatchTrueE.flatten().repeat((nPred, 1)), 0, 1))),
#         #                   0)
#         # mags = normedBatchTrueE
#         distances = torch.linalg.norm(torch.sub(batchPredCoords.repeat((nTrue, 1, 1)),
#                                                 torch.swapaxes(batchTrueCoords.repeat((nPred, 1, 1)), 0, 1)),
#                                       dim = 2)
#         # print ("NTrue", nTrue)
#         # print ("NPred", nPred)
#         # print ("dist shape", distances.shape)
#         probs = torch.sum(normedBatchTrueE*normalP(distances, gaussWidth), 0)
#         # print ("prob shape ", probs)
#         jointLogLikelihood += torch.sum(normedBatchPredE*torch.log(probs))
#         # print ("LL shape", jointLogLikelihood.shape)
#         if torch.any(torch.isnan(probs)):
#             print ("wow, shit's broken")

#     nLogL = -jointLogLikelihood

#     return nLogL

# def shapeLoss(output, truth):
#     predLLH = LLH(output, truth)
#     selfLLH = LLH(truth, truth)
#     # print ("LLHs", predLLH, selfLLH)
#     return  predLLH - selfLLH

# def normLoss(output, truth):
#     gain = 1.e1
#     outputTotal = gain*torch.sum(torch.abs(output.features[:,0]))
#     truthTotal = torch.sum(truth.features)
#     print ("total Edeps", outputTotal, truthTotal)
#     return nn.MSELoss()(outputTotal, truthTotal)

def MSE(output, truth):
    diff = (output - truth).features
    return nn.MSELoss()(diff, torch.zeros_like(diff))
