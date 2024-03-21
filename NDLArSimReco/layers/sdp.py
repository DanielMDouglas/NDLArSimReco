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
        print (inpt.features[:,0,None].shape)
        x = ME.SparseTensor(features = inpt.features[:,0,None],
                            coordinates = inpt.coordinates)
        std = inpt.features[:,1]
        print ("sdp forward inpt", x)
        print ("sdp forward std", std)
        assert len(x.shape) >= 2, x.shape  # First dimension is a batch dimension.

        # Two separate implementations:
        # the top one is for larger models and the bottom for smaller models.
        x.requires_grad_()
        y = self.net(x)
        mean = y.features[:,0]
        print (y)
        print (mean)
        jacs = []
        for i in range(self.num_outputs):
            jacs.append(torch.autograd.grad(
                mean.sum(0), x.features,
                # mean[i].sum(0), x,
                create_graph=self.create_graph, retain_graph=True
            )[0])
        jac = torch.stack(jacs, dim=1).reshape(
            x.shape[0], self.num_outputs, np.prod(x.shape[1:])
        )

        # input std is a vector, assume diagonal input covariance
        print (std.shape)
        print (jac.shape, std.shape, torch.diag(std**2).shape)
        cov_inpt = torch.diag(std**2)
        # print (torch.inner(cov_inpt, jac.transpose(-1, 0)))
        # print (torch.bmm(cov_inpt, jac.transpose(-1, -2)))
        cov = torch.inner(jac, torch.inner(cov_inpt, jac.transpose(-1, 0)))
       
        return y, cov
