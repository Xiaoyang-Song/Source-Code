import torch
import torch.nn as nn
from icecream import ic
# Credit: part of the code (1/2) is borrowed from:
# https://github.com/Michaelvll/DeepCCA/blob/master/objectives.py


class CCA(nn.Module):
    def __init__(self, outdim_size, use_all_singular_values, device):
        super().__init__()
        self.outdim_size = outdim_size
        self.use_all_singular_values = use_all_singular_values
        self.device = device
        # print(device)

    def forward(self, H1, H2):
        """
        It is the loss function of CCA as introduced in the original paper. 
        There can be other formulations.
        """

        r1 = 1e-3
        r2 = 1e-3
        eps = 1e-9

        H1, H2 = H1.t(), H2.t()
        assert torch.isnan(H1).sum().item() == 0, f'{H1}'
        assert torch.isnan(H2).sum().item() == 0, f'{H2}'
        o1 = o2 = H1.size(0)
        corr_vec = None

        m = H1.size(1)
        H1bar = H1 - H1.mean(dim=1).unsqueeze(dim=1)
        H2bar = H2 - H2.mean(dim=1).unsqueeze(dim=1)
        assert torch.isnan(H1bar).sum().item() == 0
        assert torch.isnan(H2bar).sum().item() == 0

        SigmaHat12 = (1.0 / (m - 1)) * torch.matmul(H1bar, H2bar.t())
        SigmaHat11 = (1.0 / (m - 1)) * torch.matmul(H1bar,
                                                    H1bar.t()) + r1 * torch.eye(o1, device=self.device)
        SigmaHat22 = (1.0 / (m - 1)) * torch.matmul(H2bar,
                                                    H2bar.t()) + r2 * torch.eye(o2, device=self.device)
        assert torch.isnan(SigmaHat11).sum().item() == 0
        assert torch.isnan(SigmaHat12).sum().item() == 0
        assert torch.isnan(SigmaHat22).sum().item() == 0

        # Check Covariance Matrix
        # print(SigmaHat11)
        # print(SigmaHat22)
        # Check singularity
        # print(torch.linalg.det(SigmaHat11))
        # print(torch.linalg.det(SigmaHat22))
        # Calculating the root inverse of covariance matrices by using eigen decomposition
        # [D1, V1] = torch.symeig.eigh(SigmaHat11, eigenvectors=True)
        # [D2, V2] = torch.symeig.eigh(SigmaHat22, eigenvectors=True)
        [D1, V1] = torch.linalg.eigh(SigmaHat11)
        [D2, V2] = torch.linalg.eigh(SigmaHat22)
        # print(D1)
        # print(V2)
        assert torch.isnan(D1).sum().item() == 0
        assert torch.isnan(D2).sum().item() == 0
        assert torch.isnan(V1).sum().item() == 0
        assert torch.isnan(V2).sum().item() == 0

        # Added to increase stability
        posInd1 = torch.gt(D1, eps).nonzero()[:, 0]
        D1 = D1[posInd1]
        V1 = V1[:, posInd1]
        posInd2 = torch.gt(D2, eps).nonzero()[:, 0]
        D2 = D2[posInd2]
        V2 = V2[:, posInd2]
        # print(posInd1.size())
        # print(posInd2.size())

        SigmaHat11RootInv = torch.matmul(
            torch.matmul(V1, torch.diag(D1 ** -0.5)), V1.t())
        SigmaHat22RootInv = torch.matmul(
            torch.matmul(V2, torch.diag(D2 ** -0.5)), V2.t())

        Tval = torch.matmul(torch.matmul(SigmaHat11RootInv,
                                         SigmaHat12), SigmaHat22RootInv)
        # print(Tval.size())

        if self.use_all_singular_values:
            # all singular values are used to calculate the correlation
            tmp = torch.matmul(Tval.t(), Tval)
            corr = torch.trace(torch.sqrt(tmp))
            assert torch.isnan(corr).item() == 0
        else:
            # just the top self.outdim_size singular values are used
            trace_TT = torch.matmul(Tval.t(), Tval)
            # regularization for more stability
            trace_TT = torch.add(trace_TT, (torch.eye(
                trace_TT.shape[0])*r1).to(self.device))
            # U, V = torch.symeig(trace_TT, eigenvectors=True)
            U, V = torch.linalg.eigh(trace_TT)
            U = torch.where(U > eps, U, (torch.ones(
                U.shape).double()*eps).to(self.device))
            U = U.topk(self.outdim_size)[0]
            corr_vec = torch.sqrt(U)
            corr = torch.sum(torch.sqrt(U))
        return -corr, corr_vec
