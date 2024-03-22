import torch
import numpy as np
import MinkowskiEngine as ME
ME.set_sparse_tensor_operation_mode(ME.SparseTensorOperationMode.SHARE_COORDINATE_MANAGER)

# PyTorch SDP wrapper implementation.  (tested with PyTorch version 1.13.1)


class SDP(torch.nn.Module):
    def __init__(self, net, num_outputs=1, create_graph=True):
        super(SDP, self).__init__()
        self.net = net
        self.num_outputs = num_outputs
        self.create_graph = create_graph

    def forward(self, inpt):
        x = ME.SparseTensor(features = inpt.features[:,0,None],
                            coordinates = inpt.coordinates)

        inpt_mean = inpt.features[:,0]
        inpt_std = inpt.features[:,1]

        x.features.requires_grad_()
        y = self.net(x)
        pred_mean = y.features[:,0]

        batches = torch.unique(inpt.coordinates[:,0])

        pred_means = []
        pred_std_sdp = []
        
        for batchInd in batches:
            batch_inpt_mask = inpt.coordinates[:,0] == batchInd
            batch_inpt_std = inpt.features[batch_inpt_mask,1]
            batch_inpt_st = ME.SparseTensor(features = inpt.features[batch_inpt_mask,0,None],
                                            coordinates = inpt.coordinates[batch_inpt_mask])
            batch_inpt_st.features.requires_grad_()

            batch_pred = self.net(batch_inpt_st)

            batch_pred_mean = batch_pred.features[batchInd]

            jac = torch.autograd.grad(
                batch_pred_mean,
                batch_inpt_st.features,
                create_graph=self.create_graph, retain_graph=True,
                allow_unused = True,
            )[0]

            sigma_sdp = torch.inner(jac.T, torch.inner(torch.diag(batch_inpt_std**2), jac.T).T)

            pred_means.append(batch_pred_mean)
            pred_std_sdp.append(sigma_sdp)

        pred_mean = torch.Tensor(pred_means)
        pred_std_sdp = torch.Tensor(pred_std_sdp)

        print (pred_mean.get_device())
        print (pred_std_sdp.get_device())
        print (torch.stack((pred_mean, pred_std_sdp)).get_device())
        print (y.coordinates.get_device())
        
        result = ME.SparseTensor(features = torch.stack((pred_mean, pred_std_sdp)).T,
                                 coordinates = y.coordinates)

        return result
