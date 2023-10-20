import torch
import torch.nn as nn

import numpy as np

class loss:
    def __init__(self):
        pass
    def feature_map(self, outputSparseTensor):
        """
        Each subclass implements a function which takes
        the network output sparse tensor and returns a
        tuple of features mapped to their physical meaning.
        If not specified, just pass the 0th feature
        """
        feature = outputSparseTensor.features[:,0]
        return feature,
    def truth_map(self, truth):
        truthTensor = truth.features[:,0]
        return truthTensor
    def loss(self, truth, *mappedFeatures):
        """
        Each subclass implements a loss function which
        accepts the G.T. (sparseTensor), mapped feature tensors
        (as a tuple), and returns a rank-0 tensor
        representing the loss value
        """
        return 0
    def __call__(self, output, truth):
        """
        Compose the loss function with the feature map
        """
        return self.loss(self.truth_map(truth), *self.feature_map(output))

class MSE_stock (loss): 
    def loss(self, truth, pred):
        diff = (pred - truth)
        return nn.MSELoss()(diff, torch.zeros_like(diff))

class MSE (loss):
    def loss(self, truth, pred):
        diff = (pred - truth)

        se = torch.pow(diff, 2)
        mse = torch.sum(se)/len(diff)

        return mse

class MSE_to_scalar (loss):
    def feature_map(self, outputSparseTensor):
        """
        Each subclass implements a function which takes
        the network output sparse tensor and returns a
        tuple of features mapped to their physical meaning.
        If not specified, just pass the 0th feature
        """
        return outputSparseTensor.features[:,0],
    def truth_map(self, truth):
        return truth
    def loss(self, truth, pred):
        print (truth, pred)
        return nn.MSELoss()(pred, truth)

class NLL_homog (loss):
    def loss(self, truth, mean):
        sigma = torch.ones_like(mean)

        diff = (mean - truth)
    
        logp = -0.5*torch.pow(diff/sigma, 2) - torch.log(sigma)
        # divide by number of points (to help smoothness batch to batch)
        LL = torch.sum(logp)/len(diff) 
        
        return -LL
    
class NLL (loss):
    def feature_map(self, outputSparseTensor):
        mean = outputSparseTensor.features[:,0]

        epsilon = 1.e-2
        sigma = torch.exp(outputSparseTensor.features[:,1]) + epsilon

        return mean, sigma
    def loss(self, truth, mean, sigma):    
        diff = (mean - truth)
        
        logp = -0.5*torch.pow(diff/sigma, 2) - torch.log(sigma) # + np.log(np.sqrt(2*np.pi)), ignored

        LL = torch.sum(logp)/len(diff)
        
        return -LL

class NLL_reluError (loss):
    def feature_map(self, outputSparseTensor):
        mean = outputSparseTensor.features[:,0]

        epsilon = 1.e-2 
        sigma = torch.relu(outputSparseTensor.features[:,1]) + epsilon

        return mean, sigma
    def loss(self, truth, mean, sigma):    
        diff = (mean - truth)
        logp = -0.5*torch.pow(diff/sigma, 2) - torch.log(sigma) # + np.log(np.sqrt(2*np.pi)), ignored

        LL = torch.sum(logp)/len(diff)
        
        return -LL

class NLL_reluError_masked (loss):
    def feature_map(self, outputSparseTensor):
        self.mask = outputSparseTensor.features[:,0] > 0.25
        mean = outputSparseTensor.features[:,0]
        mean = torch.where(self.mask, mean, torch.zeros_like(mean))

        epsilon = 1.e-2 
        sigma = torch.relu(outputSparseTensor.features[:,1]) + epsilon

        return mean, sigma
    def loss(self, truth, mean, sigma):    
        diff = (mean - truth)
        logp = -0.5*torch.pow(diff/sigma, 2) - torch.log(sigma) # + np.log(np.sqrt(2*np.pi)), ignored

        LL = torch.sum(logp)/len(diff)
        
        return -LL

class NLL_moyal (loss):
    def feature_map(self, outputSparseTensor):
        mean = outputSparseTensor.features[:,0]

        epsilon = 1.e-2
        # the two instances of epsilon here mean that epsilon can be adjusted during training
        # a predicted value above the previous epsilon will not change
        sigma = torch.relu(outputSparseTensor.features[:,1] - epsilon) + epsilon

        return mean, sigma
    def loss(self, truth, mean, sigma):
        y = (truth - mean)/sigma

        logp  = -1.*(y + torch.exp(-1.*y))/2 - np.log(2*np.pi)/2 - torch.log(sigma)
        
        LL = torch.sum(logp)/len(y)

        return -LL

class voxOcc (loss):
    def feature_map(self, outputSparseTensor):
        mean = outputSparseTensor.features[:,0]

        epsilon = 1.e-2 
        sigma = torch.relu(outputSparseTensor.features[:,1]) + epsilon

        isFilledVoxel = outputSparseTensor.features[:,2]
        isEmptyVoxel = outputSparseTensor.features[:,3]

        return mean, sigma, isFilledVoxel, isEmptyVoxel
    def prediction(self, outputSparseTensor):
        mean, sigma, isFilledVoxel, isEmptyVoxel = self.feature_map(outputSparseTensor)
        inferredOccupancy = torch.argmax(torch.stack([isEmptyVoxel, isFilledVoxel]).T, axis = 1)

        maskedMean = mean[inferredOccupancy]
        maskedSima = sigma[inferredOccupancy]

        return maskedMean, maskedSigma

    def occupancyLoss(self, truth, mean, sigma, isFilledVoxel, isEmptyVoxel):
        inferredOccupancy = torch.stack([isEmptyVoxel, isFilledVoxel]).T
        trueOccupancy = (truth > 0).long()
        
        return nn.CrossEntropyLoss()(inferredOccupancy, trueOccupancy)

    def loss(self, truth, mean, sigma, isFilledVoxel, isEmptyVoxel):    
        occupancyLoss = self.occupancyLoss(truth, mean, sigma, isFilledVoxel, isEmptyVoxel)
        return occupancyLoss

class NLL_voxOcc (loss):
    def feature_map(self, outputSparseTensor):
        mean = outputSparseTensor.features[:,0]

        epsilon = 1.e-2 
        sigma = torch.relu(outputSparseTensor.features[:,1]) + epsilon

        isFilledVoxel = outputSparseTensor.features[:,2]
        isEmptyVoxel = outputSparseTensor.features[:,3]

        return mean, sigma, isFilledVoxel, isEmptyVoxel
    def prediction(self, outputSparseTensor):
        mean, sigma, isFilledVoxel, isEmptyVoxel = self.feature_map(outputSparseTensor)
        inferredOccupancy = torch.argmax(torch.stack([isEmptyVoxel, isFilledVoxel]).T, axis = 1)

        maskedMean = mean[inferredOccupancy]
        maskedSima = sigma[inferredOccupancy]

        return maskedMean, maskedSigma

    def occupancyLoss(self, truth, mean, sigma, isFilledVoxel, isEmptyVoxel):
        inferredOccupancy = torch.stack([isEmptyVoxel, isFilledVoxel]).T
        trueOccupancy = (truth > 0).long()
        
        return nn.CrossEntropyLoss()(inferredOccupancy, trueOccupancy)

    def NLLLoss(self, truth, mean, sigma, isFilledVoxel, isEmptyVoxel):
        diff = (mean - truth)
        logp = -0.5*torch.pow(diff/sigma, 2) - torch.log(sigma) # + np.log(np.sqrt(2*np.pi)), ignored

        LL = torch.sum(logp)/len(diff)
        return -LL
    
    def loss(self, truth, mean, sigma, isFilledVoxel, isEmptyVoxel):    
        occupancyLoss = self.occupancyLoss(truth, mean, sigma, isFilledVoxel, isEmptyVoxel)
        NLL = self.NLLLoss(truth, mean, sigma, isFilledVoxel, isEmptyVoxel)
        
        lambd = 0.1
        return lambd*NLL + occupancyLoss

class NLL_voxOcc_masked (loss):
    def feature_map(self, outputSparseTensor):
        mean = outputSparseTensor.features[:,0]

        epsilon = 1.e-2 
        sigma = torch.relu(outputSparseTensor.features[:,1]) + epsilon

        isFilledVoxel = outputSparseTensor.features[:,2]
        isEmptyVoxel = outputSparseTensor.features[:,3]

        self.mask = outputSparseTensor.features[:,2] > outputSparseTensor.features[:,3]

        mean = torch.where(self.mask, mean, torch.zeros_like(mean))
        sigma = torch.where(self.mask, sigma, torch.ones_like(sigma))

        return mean, sigma, isFilledVoxel, isEmptyVoxel
    def prediction(self, outputSparseTensor):
        mean, sigma, isFilledVoxel, isEmptyVoxel = self.feature_map(outputSparseTensor)
        inferredOccupancy = torch.argmax(torch.stack([isEmptyVoxel, isFilledVoxel]).T, axis = 1)

        maskedMean = mean[inferredOccupancy]
        maskedSigma = sigma[inferredOccupancy]

        return maskedMean, maskedSigma

    def occupancyLoss(self, truth, mean, sigma, isFilledVoxel, isEmptyVoxel):
        inferredOccupancy = torch.softmax(torch.stack([isEmptyVoxel, isFilledVoxel]).T, axis = 1)
        trueOccupancy = (truth > 0).long()
        
        return nn.CrossEntropyLoss()(inferredOccupancy, trueOccupancy)

    def NLLLoss(self, truth, mean, sigma, isFilledVoxel, isEmptyVoxel):
        diff = (mean - truth)
        logp = -0.5*torch.pow(diff/sigma, 2) - torch.log(sigma) # + np.log(np.sqrt(2*np.pi)), ignored

        LL = torch.sum(logp)/len(diff)
        return -LL

    def loss(self, truth, mean, sigma, isFilledVoxel, isEmptyVoxel):    
        occupancyLoss = self.occupancyLoss(truth, mean, sigma, isFilledVoxel, isEmptyVoxel)
        NLL = self.NLLLoss(truth, mean, sigma, isFilledVoxel, isEmptyVoxel)
        
        # lambda = 0.1
        # return lambda*NLL + occupancyLoss
        return NLL

class NLL_voxOcc_softmax_masked (loss):
    def feature_map(self, outputSparseTensor):
        mean = outputSparseTensor.features[:,0]

        epsilon = 1.e-2 
        sigma = torch.relu(outputSparseTensor.features[:,1]) + epsilon

        isFilledVoxel = torch.sigmoid(outputSparseTensor.features[:,2])
        # isEmptyVoxel = outputSparseTensor.features[:,3]

        # self.mask = torch.softmax(torch.stack([isEmptyVoxel,
        #                                        isFilledVoxel]).T, axis = 1)

        # print (self.mask)
        # print (self.mask.shape)
        # mean = torch.sum(self.mask[0]*mean, self.mask[1]*torch.zeros_like(mean))
        # sigma = torch.sum(self.mask[0]*sigma, self.mask[1]*torch.ones_like(sigma))

        return mean, sigma, isFilledVoxel

    def prediction(self, outputSparseTensor):
        mean, sigma, isFilledVoxel, isEmptyVoxel = self.feature_map(outputSparseTensor)
        inferredOccupancy = torch.argmax(torch.stack([isEmptyVoxel, isFilledVoxel]).T, axis = 1)

        maskedMean = mean[inferredOccupancy]
        maskedSigma = sigma[inferredOccupancy]

        return maskedMean, maskedSigma

    def occupancyLoss(self, truth, mean, sigma, isFilledVoxel):
        # inferredOccupancy = torch.softmax(torch.stack([isEmptyVoxel, isFilledVoxel]).T, axis = 1)
        # trueOccupancy = torch.stack([(truth <= 0).long(),
        #                              (truth > 0).long()]).T
        inferredOccupancy = isFilledVoxel
        trueOccupancy = (truth > 0).float()
        
        # return nn.CrossEntropyLoss()(inferredOccupancy, trueOccupancy)
        return nn.functional.binary_cross_entropy(inferredOccupancy, trueOccupancy)

    def NLLLoss(self, truth, mean, sigma, isFilledVoxel):
        truthMask = truth > 0
        maskedMean = mean[truthMask]
        maskedTruth = truth[truthMask]
        maskedSigma = sigma[truthMask]

        diff = (maskedMean - maskedTruth)
        logp = -0.5*torch.pow(diff/maskedSigma, 2) - torch.log(maskedSigma) # + np.log(np.sqrt(2*np.pi)), ignored

        LL = torch.sum(logp)/sum(truthMask)
        return -LL

    def loss(self, truth, mean, sigma, isFilledVoxel):    
        occupancyLoss = self.occupancyLoss(truth, mean, sigma, isFilledVoxel)
        NLL = self.NLLLoss(truth, mean, sigma, isFilledVoxel)
        
        l = 1
        return l*NLL + occupancyLoss
        # return NLL

class MSE_voxOcc_softmax_masked (loss):
    def feature_map(self, outputSparseTensor):
        mean = outputSparseTensor.features[:,0]

        isFilledVoxel = torch.sigmoid(outputSparseTensor.features[:,1])

        return mean, isFilledVoxel

    def prediction(self, outputSparseTensor):
        mean, isFilledVoxel = self.feature_map(outputSparseTensor)
        inferredOccupancy = isFilledVoxel > 0.5

        maskedMean = mean[inferredOccupancy]

        return maskedMean

    def occupancyLoss(self, truth, mean, isFilledVoxel):
        inferredOccupancy = isFilledVoxel
        trueOccupancy = (truth > 0).float()
        
        return nn.functional.binary_cross_entropy(inferredOccupancy, trueOccupancy)

    def MSELoss(self, truth, mean, isFilledVoxel):
        truthMask = truth > 0
        maskedMean = mean[truthMask]
        maskedTruth = truth[truthMask]

        diff = (maskedMean - maskedTruth)
        MSE = torch.mean(torch.pow(diff, 2))
        return MSE

    def loss(self, truth, mean, isFilledVoxel):    
        occupancyLoss = self.occupancyLoss(truth, mean, isFilledVoxel)
        MSE = self.MSELoss(truth, mean, isFilledVoxel)
        
        l = 1
        return l*MSE + occupancyLoss
        # return NLL

class MSE_voxOcc_softmax_masked_totE (loss):
    def feature_map(self, outputSparseTensor):
        mean = outputSparseTensor.features[:,0]

        isFilledVoxel = torch.sigmoid(outputSparseTensor.features[:,1])

        return mean, isFilledVoxel

    def prediction(self, outputSparseTensor):
        mean, isFilledVoxel = self.feature_map(outputSparseTensor)
        inferredOccupancy = isFilledVoxel > 0.5

        maskedMean = mean[inferredOccupancy]

        return maskedMean

    def energyLoss(self, truth, mean, isFilledVoxel):
        truthMask = truth > 0
        maskedMean = torch.sum(mean[truthMask])
        maskedTruth = torch.sum(truth[truthMask])

        diff = (maskedMean - maskedTruth)
        MSE = torch.mean(torch.pow(diff, 2))
        return MSE

    def occupancyLoss(self, truth, mean, isFilledVoxel):
        inferredOccupancy = isFilledVoxel
        trueOccupancy = (truth > 0).float()
        
        return nn.functional.binary_cross_entropy(inferredOccupancy, trueOccupancy)

    def MSELoss(self, truth, mean, isFilledVoxel):
        truthMask = truth > 0
        maskedMean = mean[truthMask]
        maskedTruth = truth[truthMask]

        diff = (maskedMean - maskedTruth)
        MSE = torch.mean(torch.pow(diff, 2))
        return MSE

    def loss(self, truth, mean, isFilledVoxel):    
        occupancyLoss = self.occupancyLoss(truth, mean, isFilledVoxel)
        MSE = self.MSELoss(truth, mean, isFilledVoxel)
        energyLoss = self.energyLoss(truth, mean, isFilledVoxel)
        
        l = 1
        return l*MSE + occupancyLoss + energyLoss
        # return NLL

class NLL_voxOcc_softmax_masked_inference (loss):
    def feature_map(self, outputSparseTensor):
        mean = outputSparseTensor.features[:,0]

        epsilon = 1.e-2 
        sigma = torch.relu(outputSparseTensor.features[:,1]) + epsilon

        isFilledVoxel = torch.sigmoid(outputSparseTensor.features[:,2])
        # isEmptyVoxel = outputSparseTensor.features[:,3]

        self.mask = isFilledVoxel>0.5

        # print (self.mask)
        # print (self.mask.shape)
        mean = torch.where(self.mask, mean, torch.zeros_like(mean))
        sigma = torch.where(self.mask, sigma, torch.ones_like(sigma))

        return mean, sigma, isFilledVoxel

    def prediction(self, outputSparseTensor):
        mean, sigma, isFilledVoxel = self.feature_map(outputSparseTensor)
        # inferredOccupancy = torch.argmax(torch.stack([isEmptyVoxel, isFilledVoxel]).T, axis = 1
        inferredOccupancy = isFilledVoxel > 0.5

        maskedMean = mean[inferredOccupancy]
        maskedSigma = sigma[inferredOccupancy]

        return maskedMean, maskedSigma

    def occupancyLoss(self, truth, mean, sigma, isFilledVoxel):
        # inferredOccupancy = torch.softmax(torch.stack([isEmptyVoxel, isFilledVoxel]).T, axis = 1)
        # trueOccupancy = torch.stack([(truth <= 0).long(),
        #                              (truth > 0).long()]).T
        inferredOccupancy = isFilledVoxel
        trueOccupancy = (truth > 0).float()
        
        # return nn.CrossEntropyLoss()(inferredOccupancy, trueOccupancy)
        return nn.functional.binary_cross_entropy(inferredOccupancy, trueOccupancy)

    def NLLLoss(self, truth, mean, sigma, isFilledVoxel):
        diff = (mean - truth)
        logp = -0.5*torch.pow(diff/sigma, 2) - torch.log(sigma) # + np.log(np.sqrt(2*np.pi)), ignored

        LL = torch.sum(logp)/len(diff)
        return -LL

    def loss(self, truth, mean, sigma, isFilledVoxel):    
        # occupancyLoss = self.occupancyLoss(truth, mean, sigma, isFilledVoxel)
        NLL = self.NLLLoss(truth, mean, sigma, isFilledVoxel)
        
        # l = 1
        # return l*NLL + occupancyLoss
        return NLL

class CrossEntropy (loss):
    def feature_map(self, outputTensor):
        return outputTensor.features,

    def truth_map(self, truth):
        return truth
        
    def loss(self, truth, output):
        return nn.CrossEntropyLoss()(output, truth)

class semanticSegmentationCrossEntropy (loss):
    def feature_map(self, outputTensor):
        return outputTensor.features[:,0],

    def truth_map(self, truth):
        return truth.features[:,0]
        
    def loss(self, truth, output):
        return torch.nn.functional.binary_cross_entropy_with_logits(output, truth)
