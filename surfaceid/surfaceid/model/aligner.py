import numpy as np
import torch
import torch.nn as nn

RLIM = 1.5

class Aligner1(nn.Module):
    def __init__(self, pos0, pos, mask, nangle=16, device="cpu", npairs=0, float_type=np.float32):
        """_summary_

        :param pos0: _description_
        :type pos0: _type_
        :param pos: _description_
        :type pos: _type_
        :param mask: _description_
        :type mask: _type_
        :param nangle: _description_, defaults to 16
        :type nangle: int, optional
        :param device: _description_, defaults to "cpu"
        :type device: str, optional
        :param npairs: _description_, defaults to 0
        :type npairs: int, optional
        :param float_type: _description_, defaults to np.float32
        :type float_type: _type_, optional
        """
        super(Aligner1, self).__init__()
        angle_range = 2 * np.pi
        dangle = angle_range / (nangle - 1)
        ts = np.arange(0, angle_range + dangle, dangle, dtype=float_type)
        alphas, betas, gammas = np.meshgrid(ts, ts, ts)
        alphas = alphas.flatten()
        shape = (len(alphas), 1, 1)
        self.alphas = nn.parameter.Parameter(
            torch.tensor(alphas.reshape(shape)))
        self.betas = nn.parameter.Parameter(
            torch.tensor(betas.flatten().reshape(shape)))
        self.gammas = nn.parameter.Parameter(
            torch.tensor(gammas.flatten().reshape(shape)))
        self.Tx = nn.parameter.Parameter(
            torch.tensor(np.zeros(shape, dtype=float_type)))
        self.Ty = nn.parameter.Parameter(
            torch.tensor(np.zeros(shape, dtype=float_type)))
        self.Tz = nn.parameter.Parameter(
            torch.tensor(np.zeros(shape, dtype=float_type)))

        pos = pos.astype(float_type)
        pos0 = pos0.astype(float_type)
        n0 = len(pos0)
        self.x0 = torch.tensor(pos0[:, 0].reshape(1, n0, 1)).to(device)
        self.y0 = torch.tensor(pos0[:, 1].reshape(1, n0, 1)).to(device)
        self.z0 = torch.tensor(pos0[:, 2].reshape(1, n0, 1)).to(device)
        n = len(pos)
        self.x = torch.tensor(pos[:, 0].reshape(1, 1, n)).to(device)
        self.y = torch.tensor(pos[:, 1].reshape(1, 1, n)).to(device)
        self.z = torch.tensor(pos[:, 2].reshape(1, 1, n)).to(device)
        self.mask = mask.reshape(1, n0, n).to(device)
        self.npairs = npairs

        return None

    def forward(self):
        # transform
        ca, cb, cg = torch.cos(self.alphas), torch.cos(
            self.betas), torch.cos(self.gammas)
        sa, sb, sg = torch.sin(self.alphas), torch.sin(
            self.betas), torch.sin(self.gammas)
        x = self.x
        y = self.y
        z = self.z
        x1 = ca * cb * x + (ca * sb * sg - sa * cg) * y + \
            (ca * sb * cg + sa * sg) * z + self.Tx
        y1 = sa * cb * x + (sa * sb * sg + ca * cg) * y + \
            (sa * sb * cg - ca * sg) * z + self.Ty
        z1 = -sb * x + cb * sg * y + cb * cg * z + self.Tz
        r = ((x1 - self.x0).square() + (y1 - self.y0).square() +
             (z1 - self.z0).square() + 1e-6).sqrt()
        losses = (r * self.mask).sum([1, 2]) / self.npairs
        loss = losses.sum()

        return loss, losses


def get_r(xyz0, xyz):
    """_summary_

    :param xyz0: _description_
    :type xyz0: _type_
    :param xyz: _description_
    :type xyz: _type_
    :return: _description_
    :rtype: _type_
    """
    r = np.sqrt(
        np.sum(np.square(xyz0[:, None, :] - xyz[None, :, :]), axis=-1) + 1e-6)
    return r


def get_score1(xyz0, xyz, mask, npairs):
    """_summary_

    :param xyz0: _description_
    :type xyz0: _type_
    :param xyz: _description_
    :type xyz: _type_
    :param mask: _description_
    :type mask: _type_
    :param npairs: _description_
    :type npairs: _type_
    :return: _description_
    :rtype: _type_
    """
    r = get_r(xyz0, xyz)
    loss = np.sum(r * mask) / npairs

    return loss


class Aligner2(nn.Module):
    """_summary_

    :param nn: _description_
    :type nn: _type_
    """
    def __init__(self, pos0, f0, pos, f, sig=2.5, nangle=16, gamma=100, device="cpu", float_type=np.float32):
        super(Aligner2, self).__init__()
        angle_range = np.pi / 6.0
        ts = np.arange(-angle_range, angle_range, 2 *
                       angle_range / nangle, dtype=float_type)
        alphas, betas, gammas = np.meshgrid(ts, ts, ts)
        alphas = alphas.flatten()
        shape = (len(alphas), 1, 1)
        self.alphas = nn.parameter.Parameter(
            torch.tensor(alphas.reshape(shape)))
        self.betas = nn.parameter.Parameter(
            torch.tensor(betas.flatten().reshape(shape)))
        self.gammas = nn.parameter.Parameter(
            torch.tensor(gammas.flatten().reshape(shape)))
        self.Tx = nn.parameter.Parameter(
            torch.tensor(np.zeros(shape, dtype=float_type)))
        self.Ty = nn.parameter.Parameter(
            torch.tensor(np.zeros(shape, dtype=float_type)))
        self.Tz = nn.parameter.Parameter(
            torch.tensor(np.zeros(shape, dtype=float_type)))

        pos = pos.astype(float_type)
        pos0 = pos0.astype(float_type)
        f = f.astype(float_type)
        f0 = f0.astype(float_type)
        n0 = len(pos0)
        self.x0 = torch.tensor(pos0[:, 0].reshape(1, 1, n0)).to(device)
        self.y0 = torch.tensor(pos0[:, 1].reshape(1, 1, n0)).to(device)
        self.z0 = torch.tensor(pos0[:, 2].reshape(1, 1, n0)).to(device)
        n = len(pos)
        self.x = torch.tensor(pos[:, 0].reshape(1, n, 1)).to(device)
        self.y = torch.tensor(pos[:, 1].reshape(1, n, 1)).to(device)
        self.z = torch.tensor(pos[:, 2].reshape(1, n, 1)).to(device)
        self.f_matrix = torch.tensor(f.reshape(
            1, n, 1, 5) - f0.reshape(1, 1, n0, 5)).square().sum(dim=-1).to(device)
        self.sig = sig
        self.gamma = gamma

        return

    def forward(self):
        # transform
        ca, cb, cg = torch.cos(self.alphas), torch.cos(
            self.betas), torch.cos(self.gammas)
        sa, sb, sg = torch.sin(self.alphas), torch.sin(
            self.betas), torch.sin(self.gammas)
        x = self.x
        y = self.y
        z = self.z
        x1 = ca * cb * x + (ca * sb * sg - sa * cg) * y + \
            (ca * sb * cg + sa * sg) * z + self.Tx
        y1 = sa * cb * x + (sa * sb * sg + ca * cg) * y + \
            (sa * sb * cg - ca * sg) * z + self.Ty
        z1 = - sb * x + cb * sg * y + cb * cg * z + self.Tz
        r = ((x1 - self.x0).square() +
             (y1 - self.y0).square() + (z1 - self.z0).square())
        r = r.sqrt()  # [Nmodels, N, N0]
        mask = (r < RLIM).float()
        r = 100. * (1. - mask) + r * mask
        nactive = (mask.sum(1) > 0).sum(1)  # npoints in query close to target
        # effectively select one target point per reference point
        w = torch.softmax(-r / self.sig, dim=1)
        losses = (w * mask * self.f_matrix).sum((1, 2)) / nactive
        losses_ = losses + self.gamma * \
            (self.Tx.square() + self.Ty.square() + self.Tz.square())
        loss = losses_.sum()

        return loss, losses