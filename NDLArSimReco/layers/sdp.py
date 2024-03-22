import torch
import numpy as np
import MinkowskiEngine as ME
ME.set_sparse_tensor_operation_mode(ME.SparseTensorOperationMode.SHARE_COORDINATE_MANAGER)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

        pred_sdp_var = []

        for i in batches:
            jac = torch.autograd.grad(
                pred_mean[i],
                x.features,
                create_graph=self.create_graph, retain_graph=True,
                allow_unused = True,
            )[0]

            nInputs = x.coordinates[:,0] == i
            mask = x.coordinates[:,0] == i

            batch_inpt_std = inpt_std[mask]
            batchJac = jac[mask]

            pred_sdp_var.append(torch.inner(batchJac.T,
                                            torch.inner(torch.diag(batch_inpt_std**2),
                                                        batchJac.T).T))

        pred_sdp_var = torch.Tensor(pred_sdp_var).to(device)
            
        result = ME.SparseTensor(features = torch.stack((pred_mean, pred_sdp_var)).T,
                                 coordinates = y.coordinates)

        return result
