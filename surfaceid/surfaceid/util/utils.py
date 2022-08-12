# ** need to change var names
# ** id for each match for easier search
# ** pdb file
import seaborn as sns
from time import time
import os
import subprocess
import numpy as np
import pandas as pd
import sys
from functools import partial
import tempfile
from multiprocessing import Pool, cpu_count
from Bio.PDB import PDBParser, rotaxis2m, PDBIO
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch_scatter
import torch.optim as optim
import sys
from surfaceid.model.model import Model, clean_tensor, Mol, DNEG, DPOS, OUTDIR
from plyfile import PlyData, PlyElement
from copy import deepcopy
from scipy.special import softmax as scipy_softmax
from sklearn.decomposition import PCA
torch.set_num_threads(cpu_count())
RLIM = 1.5
SIG = 0.25
SIG_NORMAL = 3.0
# Operation
CONTACT = False
DESC = False
SEARCH = False
ALIGN = True


def gen_descriptors_contact(p1s, p2s, model, device, rho_max, outdir, fname):
    """_summary_

    :param p1s: _description_
    :type p1s: _type_
    :param p2s: _description_
    :type p2s: _type_
    :param model: _description_
    :type model: _type_
    :param device: _description_
    :type device: _type_
    :param rho_max: _description_
    :type rho_max: _type_
    :param outdir: _description_
    :type outdir: _type_
    :param fname: _description_
    :type fname: _type_
    :return: _description_
    :rtype: _type_
    """
    outs = []
    idxs = []
    pdbss = []
    bs = 256
    with torch.set_grad_enabled(False):
        for i, (p1, p2) in enumerate(zip(p1s, p2s)):
            # p1 -> p2
            print(i, p1, p2)
            x_, rho_, theta_, mask_, idxs0_, pdbs_ = get_batch(
                device, [(p1, p2)], rho_max, contacts=True, outdir=outdir)
            for i1 in range(0, len(x_), bs):
                i2 = min(i1 + bs, len(x_))
                o, _, _, _ = model(x_[i1:i2], rho_[i1:i2], theta_[
                                   i1:i2], mask_[i1:i2], calc_loss=False)
                outs.append(o.detach().cpu().numpy())
            idxs.append(idxs0_)
            pdbss.append(pdbs_)
            if p1 != p2:
                # p2 -> p1
                print(i, p2, p1)
                x_, rho_, theta_, mask_, idxs0_, pdbs_ = get_batch(
                    device, [(p2, p1)], rho_max, contacts=True, outdir=outdir)
                for i1 in range(0, len(x_), bs):
                    i2 = min(i1 + bs, len(x_))
                    o, _, _, _ = model(x_[i1:i2], rho_[i1:i2], theta_[
                                       i1:i2], mask_[i1:i2], calc_loss=False)
                    outs.append(o.detach().cpu().numpy())
                idxs.append(idxs0_)
                pdbss.append(pdbs_)
    x_desc = np.concatenate(outs)
    idxs_desc = np.concatenate(idxs)
    pdbs = np.concatenate(pdbss)
    np.savez(fname, x=x_desc, pdbs=pdbs, idxs=idxs_desc)

    return pdbs, x_desc, idxs_desc


def transform_library(x, y, z, xyz0_mean, xyz_mean, align_model, idx, n=None):
    # --- transform coord
    a, b, g = align_model.alphas[idx].item(
    ), align_model.betas[idx].item(), align_model.gammas[idx].item()
    Tx = align_model.Tx[idx].item()
    Ty = align_model.Ty[idx].item()
    Tz = align_model.Tz[idx].item()
    xyz = transform(x, y, z, a, b, g, Tx, Ty, Tz, xyz0_mean, xyz_mean)
    if n is not None:
        # Calculate normal match
        n = inverse_rotation(n[:, 0], n[:, 1], n[:, 2], a, b, g)

    return xyz, n


class Aligner1(nn.Module):
    def __init__(self, pos0, pos, mask, nangle=16, device="cpu", npairs=0, float_type=np.float32):
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
        z1 = -sb * x + cb * sg * y + cb * cg * z + self.Tz
        r = ((x1 - self.x0).square() + (y1 - self.y0).square() +
             (z1 - self.z0).square() + 1e-6).sqrt()
        losses = (r * self.mask).sum([1, 2]) / self.npairs
        loss = losses.sum()

        return loss, losses


def get_r(xyz0, xyz):
    r = np.sqrt(
        np.sum(np.square(xyz0[:, None, :] - xyz[None, :, :]), axis=-1) + 1e-6)
    return r


def get_score1(xyz0, xyz, mask, npairs):
    r = get_r(xyz0, xyz)
    loss = np.sum(r * mask) / npairs

    return loss


class Aligner2(nn.Module):
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
        # print(w.sum(dim=1), w.shape, self.x.shape)
        # assert False
        losses = (w * mask * self.f_matrix).sum((1, 2)) / nactive
        losses_ = losses + self.gamma * \
            (self.Tx.square() + self.Ty.square() + self.Tz.square())
        loss = losses_.sum()

        return loss, losses


def get_score2_helper(xyz0, fs_raw0, n0, xyz, fs_raw, n, sig, sig_normal=SIG_NORMAL):
    r = get_r(xyz0, xyz)  # [N0, N]
    mask = (r < RLIM).astype(np.float32)
    r = 100. * (1. - mask) + r * mask
    nactive = np.sum(mask.sum(axis=0) > 0)
    w = scipy_softmax(-r**2 / sig**2, axis=0)
    f_matrix = np.sum(
        np.square(fs_raw0[:, None, :] - fs_raw[None, :, :]), axis=-1)
    score = (w * mask * f_matrix).sum() / nactive
    # -- using mesh normals to compute
    # mask = (r < 0.5).astype(np.float32)
    # r = 100. * (1. - mask) + r * mask
    # nactive = np.sum(mask.sum(axis=0) > 0)
    # w = scipy_softmax(-r**2 / sig**2, axis=0)
    n_matrix = np.sum(n0[:, None, :] * n[None, :, :], axis=-1)
    # sig_normal = 2.
    # w_normal = np.exp(-r**2 / sig_normal**2)
    # n_score = (w_normal * mask * n_matrix).sum() / w_normal.sum()
    n_score = (w * mask * n_matrix).sum() / \
        nactive  # normals are already smoothed
    # print(nactive, score, n_score)
    # iselect = mask.sum(axis=1) > 0
    # n0_pca = get_pca_normal(xyz0[iselect])
    # # n0_pca = np.sign(np.sum(n0_pca * n0[iselect])) * n0_pca
    # ii, jj = np.where(mask==1)

    return nactive, score, n_score


def get_score2(xyz0, fs_raw0, n0, xyz, fs_raw, n, sig):
    nactive1, score1, n0_score = get_score2_helper(
        xyz0, fs_raw0, n0, xyz, fs_raw, n, sig)
    nactive2, score2, n_score = get_score2_helper(
        xyz, fs_raw, n, xyz0, fs_raw0, n0, sig)
    nactive = (nactive1 + nactive2) / 2.
    n_score = (n_score + n0_score) / 2.
    score = (score1 + score2) / 2.
    # # -- using planes
    # n_score = np.dot(n0_pca, n_pca)

    return nactive, n_score, score


def get_pca_normal(xyz):
    pca = PCA(n_components=3)
    pca.fit(xyz)
    q1, q2, q3 = pca.components_
    q3 = np.sign(np.dot(np.cross(q1, q2), q3)) * q3
    return q3


def get_fs(query, pdbs_desc, x_desc, idxs_desc):
    iselect = pdbs_desc == query

    return x_desc[iselect], idxs_desc[iselect]


def get_xyz_npz(fname, idxs=None):
    data = np.load(fname)
    vs = data["pos"]
    return vs[idxs]


def get_normal_npz(fname, idxs=None):
    data = np.load(fname)
    vs = data["normals"]
    return vs[idxs]


def get_xyz(plydata, idxs=None):
    vs = plydata["vertex"]
    x, y, z = vs["x"], vs["y"], vs["z"]
    if idxs is not None:
        return np.stack([x, y, z]).T[idxs]
    else:
        return x, y, z


def get_fs_npz(fname, idxs):
    x = np.load(fname)["x_initial"][idxs]
    x = np.maximum(x, -1.0)
    x = np.minimum(x, 1.0)

    return x


def get_centroid(xyz):
    return np.mean(xyz, axis=0)


def transform(x, y, z, a, b, g, Tx, Ty, Tz, xyz0_mean, xyz_mean):
    ca, cb, cg = np.cos(a), np.cos(b), np.cos(g)
    sa, sb, sg = np.sin(a), np.sin(b), np.sin(g)
    x -= xyz_mean[0]
    y -= xyz_mean[1]
    z -= xyz_mean[2]
    x1 = ca * cb * x + (ca * sb * sg - sa * cg) * y + \
        (ca * sb * cg + sa * sg) * z + Tx
    y1 = sa * cb * x + (sa * sb * sg + ca * cg) * y + \
        (sa * sb * cg - ca * sg) * z + Ty
    z1 = - sb * x + cb * sg * y + cb * cg * z + Tz
    xyz = np.stack([x1, y1, z1]).T
    xyz += xyz0_mean

    return xyz


def inverse_rotation(x, y, z, a, b, g):
    ca, cb, cg = np.cos(a), np.cos(b), np.cos(g)
    sa, sb, sg = np.sin(a), np.sin(b), np.sin(g)
    x1 = ca * cb * x + sa * cb * y - sb * z
    y1 = (ca * sb * sg - sa * cg) * x + \
        (sa * sb * sg + ca * cg) * y + cb * sg * z
    z1 = (ca * sb * cg + sa * sg) * x + \
        (sa * sb * cg - ca * sg) * y + cb * cg * z
    xyz = np.stack([x1, y1, z1]).T

    return xyz


def mean_normal(n):
    n = np.mean(n, axis=0)
    return n / np.sqrt(np.sum(n**2))


class Mol_slim:
    """_summary_
    """

    def __init__(self, device, p, contacts=True, outdir=OUTDIR):
        p1, p2 = p.split(".")
        self.p = f"{p1}.{p2}"
        try:
            fname = os.path.join(outdir, p1) + "_surface.npz"
            x = np.load(fname)
            self.list_indices = torch.tensor(
                x["list_indices"], dtype=torch.long).to(device)
            self.rho = torch.tensor(x["rho"]).to(device)
            if contacts:
                idxs = np.load(os.path.join(outdir, f"{p1}_contacts.{p2}.npy"))
            else:
                idxs = np.arange(len(x["x_initial"]))
            idxs = np.unique(idxs)
            self.idxs = idxs
            edges = x["edge_index"]
            edges = edges[:, np.all(np.isin(edges, idxs), axis=0)]
            edges = np.unique(edges, axis=1)
            edges = np.unique(np.concatenate(
                [edges, edges[::-1, :]], axis=1), axis=1)
            reorder_map = np.full(np.max(idxs)+1, -1, dtype=np.int32)
            reorder_map[idxs] = range(len(idxs))
            self.edges = torch.tensor(
                reorder_map[edges], dtype=torch.long).to(device)
            self.status = "good"
        except:
            self.status = "bad"


class Mol_with_epitopes(Mol):
    """_summary_

    :param Mol: _description_
    :type Mol: _type_
    """

    def __init__(self, p, contacts=True, outdir=OUTDIR):
        p1, p2 = p
        fname = os.path.join(outdir, p1) + "_surface.npz"
        print(fname)
        super().__init__(fname)
        self.p = f"{p1}.{p2}"
        if contacts:
            self.idxs0 = np.load(os.path.join(
                outdir, f"{p1}_contacts.{p2}.npy"))
        else:
            self.idxs0 = np.arange(len(self.x))

        # Make patches
        self.x = self.x[self.list_indices[self.idxs0]]
        self.rho = self.rho[self.idxs0]
        self.theta = self.theta[self.idxs0]
        self.mask = self.mask[self.idxs0]


def get_desc_aux(pdbs):
    """_summary_

    :param pdbs: _description_
    :type pdbs: _type_
    :return: _description_
    :rtype: _type_
    """
    n = len(np.unique(pdbs))
    starts = np.zeros(n, dtype=np.int32)
    ends = np.zeros(n, dtype=np.int32)
    idx = 0
    pdb_pivot = pdbs[0]
    pdbs_unique = []
    i = 0
    while i < (len(pdbs)-1):
        i += 1
        if pdb_pivot != pdbs[i]:
            pdbs_unique.append(pdb_pivot)
            pdb_pivot = pdbs[i]
            ends[idx] = i
            idx += 1
            starts[idx] = i
    ends[-1] = i+1
    pdbs_unique.append(pdb_pivot)
    pdbs_unique = np.asarray(pdbs_unique)
    patch_sizes = ends-starts

    return patch_sizes, pdbs_unique, starts, ends


def search(o1, rho, list_indices, within, library, library_within, pdbs_unique, patch_sizes,
           pdbs, x_desc, idxs_desc, mini_bs, device, nmin_pts, expand_radius, nmin_pts_library,
           thres, target, idxs_contact):
    summary = []
    with torch.set_grad_enabled(False):
        for lb, within_lb in zip(library, library_within):
            js = np.where(pdbs_unique == lb)[0]
            if len(js) != 1:
                # assert False
                continue
            j = js[0]
            if patch_sizes[j] < nmin_pts:
                continue
            st = time()
            o2, _ = get_fs(lb, pdbs, x_desc, idxs_desc)
            n2 = len(o2)
            ds = []
            for i1 in range(0, n2, mini_bs):
                i2 = min(i1+mini_bs, n2)
                o2_ = torch.tensor(o2[i1:i2]).to(
                    device).view(1, i2-i1, o2.shape[-1])
                d = torch.norm(o1 - o2_, dim=2)
                ds.append(d)
            d = torch.cat(ds, dim=1)
            mask = (d < thres).float().cpu().numpy()
            if idxs_contact is not None:
                # mask out non-contact points
                iselect = ~np.in1d(np.arange(len(mask)), idxs_contact)
                mask[iselect] = 0.
            hits = np.sum(mask, axis=1) > 0
            hits = np.where(hits)[0]
            nhits = len(hits)
            # get lb connectivity / hits
            if nhits >= nmin_pts:
                groups = find_clusters(
                    hits, within, rho, list_indices, nmin_pts, expand_radius)
                if groups:
                    d = d.cpu().numpy()
                    for target_nhits, target_nexpanded, idxs in groups:
                        # Find the largest library cluster (1st element in the sorted list)
                        hits_lb = np.sum(mask[idxs], axis=0) > 0
                        hits_lb = np.where(hits_lb)[0]
                        # No expanding library epitope
                        idxs_lb = find_clusters(
                            hits_lb, within_lb, None, None, nmin_pts=nmin_pts_library, expand_radius=0.0)
                        if len(idxs_lb) == 0:
                            continue
                        else:
                            lb_nhits, lb_nexpanded, idxs_lb = idxs_lb[0]
                        mean_desc_dist = np.mean(d[idxs[:, None], idxs_lb])
                        frac_lb_hits = lb_nhits / lb_nexpanded
                        summary.append((target, lb, target_nhits, target_nexpanded,
                                       mean_desc_dist, lb_nhits, lb_nexpanded, frac_lb_hits, idxs, idxs_lb))
                    print(f"{j} {lb} / dt = {time() - st:.3f}")

    return summary


def get_batch(device, ps, rho_max, contacts=True, outdir=None):
    xs = [Mol_with_epitopes(p, contacts, outdir) for p in ps]
    xs = [x for x in xs if x.status == "good"]
    patches = [(x.x, x.rho, x.theta, x.mask, x.idxs0,
                np.full(len(x.idxs0), x.p)) for x in xs]
    x, rho, theta, mask, idxs0, pdbs = zip(*patches)
    x_ = np.concatenate(x)
    rho_ = np.concatenate(rho)
    theta_ = np.concatenate(theta)
    mask_ = np.concatenate(mask)
    idxs0 = np.concatenate(idxs0)
    pdbs = np.concatenate(pdbs)

    # reduce data
    mask_[rho_ > rho_max] = 0.
    m = mask_.sum(axis=0)
    npoints = 0
    while m[npoints] > 0:
        npoints += 1
        if npoints == 200:
            break
    B = len(x)
    x_ = x_[:, :npoints]
    rho_ = rho_[:, :npoints]
    theta_ = theta_[:, :npoints]
    mask_ = mask_[:, :npoints]

    x_, rho_, theta_, mask_ = clean_tensor(device, x_, rho_, theta_, mask_)

    return x_, rho_, theta_, mask_, idxs0, pdbs


def get_whole_molecule_desc(target, model, rho_max, bs=256, device="cpu", outdir=None):
    """_summary_

    :param target: _description_
    :type target: _type_
    :param model: _description_
    :type model: _type_
    :param rho_max: _description_
    :type rho_max: _type_
    :param bs: _description_, defaults to 256
    :type bs: int, optional
    :param device: _description_, defaults to "cpu"
    :type device: str, optional
    :param outdir: _description_, defaults to None
    :type outdir: _type_, optional
    :return: _description_
    :rtype: _type_
    """
    x_, rho_, theta_, mask_, idxs0_, pdbs_ = get_batch(
        device, [(target, "none")], rho_max, contacts=False, outdir=outdir)
    outs = []
    for i1 in range(0, len(x_), bs):
        i2 = min(i1 + bs, len(x_))
        o, _, _, _ = model(x_[i1:i2], rho_[i1:i2], theta_[
                           i1:i2], mask_[i1:i2], calc_loss=False)
        outs.append(o.detach().cpu().numpy())
    o = np.concatenate(outs)
    p = Mol_slim(device, f"{target}.none", contacts=False, outdir=outdir)

    return o, p


def find_clusters(hits, within, rho, list_indices, nmin_pts=200, expand_radius=3.0):
    """ """
    taken = np.zeros_like(hits, dtype=bool)
    ii = np.arange(len(taken), dtype=np.int32)
    groups = []
    while not np.all(taken):
        hits_ = hits[~taken]
        ii_ = ii[~taken]
        idx_pivot = hits_[0]
        zs = within[idx_pivot, hits_]
        # get neighbors
        neighbors = hits_[zs]
        # update taken
        taken[ii_[zs]] = True
        groups.append(neighbors)

    # for i in range(0, len(groups), 1):
    ngroups = len(groups)
    for _ in range(20):
        i = 0
        while i < (len(groups)-1):
            group1 = groups[i]
            tmp = [group1]
            for j in range(len(groups)-1, i, -1):
                # print(i, j)
                group2 = groups[j]
                # Any hits?
                if np.any(within[group1[:, None], group2]):  # within is symmetric
                    # assert False
                    tmp.append(group2)
                    del groups[j]
            groups[i] = np.concatenate(tmp)
            i += 1
        if len(groups) == ngroups:
            break
        else:
            ngroups = len(groups)

    # eliminate low count groups
    groups = [x for x in groups if len(x) >= nmin_pts]
    groups = sorted(groups, key=lambda x: -len(x))

    groups_expanded = []
    for group in groups:
        nhits = len(group)
        if expand_radius < 0.1:
            group = np.where(within[group].sum(0) > 0)[0]
        else:
            rho_ = rho[group]
            group = np.unique(list_indices[group][(
                rho_ > 0) & (rho_ < expand_radius)])
        groups_expanded.append((nhits, len(group), group))

    return groups_expanded


def get_within(rho, list_indices, neighbor_dist):
    # Get inclusion matrix
    npts = len(rho)
    within = np.zeros((npts, npts), dtype=bool)
    for i in range(npts):
        for j, d in zip(list_indices[i], rho[i]):
            if (d <= neighbor_dist) and (d > 0):
                within[i, j] = True
            else:
                break
    return within


def get_reordered_subset(idxs, rho, list_indices):
    """returns re-ordered subset of points"""
    # idxs is the subset vertices
    idxs_new = np.arange(len(idxs), dtype=np.int32)
    reorder_map = np.full(np.max(list_indices)+1, -1, dtype=np.int32)
    reorder_map[idxs] = idxs_new
    rho_subset, list_indices_subset = rho[idxs], reorder_map[list_indices[idxs]]

    return idxs_new, rho_subset, list_indices_subset


def smooth_normals(n, rho, list_indices, sig_normal=SIG_NORMAL):
    n_smoothed = np.zeros_like(n)
    rlim = SIG_NORMAL * 3
    for i, (li, r) in enumerate(zip(list_indices, rho)):
        iselect = r < rlim
        w = np.exp(-r[iselect]**2 / sig_normal**2)
        n_smoothed[i] = np.sum(
            n[li[iselect]] * w.reshape(iselect.sum(), 1), axis=0) / w.sum()
    n_smoothed = n_smoothed / \
        np.sqrt(np.sum(np.square(n_smoothed), axis=1).reshape(len(n), 1))

    return n_smoothed


def compute_aux_vars(p1, p2, OUTDIR, expand_radius, neighbor_dist, mode,
                     contact_thres1, contact_thres2, device, smooth=True):
    """p1, p2 are binding partners. Compute smoothed normals,
    contact region indices, and a matrix that tells which points are connected
    """
    data1 = np.load(os.path.join(OUTDIR, f"{p1}_surface.npz"))
    x1 = data1["pos"]
    li1 = data1["list_indices"]
    rho1 = data1["rho"]
    n1 = data1["normals"]
    data2 = np.load(os.path.join(OUTDIR, f"{p2}_surface.npz"))
    x2 = data2["pos"]
    li2 = data2["list_indices"]
    rho2 = data2["rho"]
    n2 = data2["normals"]
    # ---- Create smoothed normals
    if smooth:
        fname1 = os.path.join(OUTDIR, f"{p1}_smoothed_normals.npy")
        n1_smoothed = smooth_normals(n1, rho1, li1)
        print(np.sum(n1 * n1_smoothed) / len(n1))
        np.save(fname1, n1_smoothed)
        fname2 = os.path.join(OUTDIR, f"{p2}_smoothed_normals.npy")
        n2_smoothed = smooth_normals(n2, rho2, li2)
        print(np.sum(n2 * n2_smoothed) / len(n2))
        np.save(fname2, n2_smoothed)
    # ---- calc dist
    x1 = torch.tensor(x1.reshape(len(x1), 1, 3)).to(device)
    x2 = torch.tensor(x2.reshape(1, len(x2), 3)).to(device)
    d = torch.norm(x1-x2, dim=2).cpu().numpy()
    if mode == "iface":
        # ---- based on iface
        ii1, ii2 = np.where(d < contact_thres1)
        if1 = np.where(data1["iface"] == 1)[0]
        if2 = np.where(data2["iface"] == 1)[0]
        # filter by iface
        ii1 = ii1[np.in1d(ii1, if1)]
        ii2 = ii2[np.in1d(ii2, if2)]
        # # ---- based on dist
        ii1_, ii2_ = np.where(d < contact_thres2)
        # combine
        ii1 = np.unique(np.concatenate([ii1, ii1_]))
        ii2 = np.unique(np.concatenate([ii2, ii2_]))
    else:
        ii1, ii2 = np.where(d < contact_thres1)
    # ---- expand
    rho1_ = rho1[ii1]
    ii1 = np.unique(li1[ii1][(rho1_ > 0) & (rho1_ < expand_radius)])
    rho2_ = rho2[ii2]
    ii2 = np.unique(li2[ii2][(rho2_ > 0) & (rho2_ < expand_radius)])
    if mode == "iface":
        print(p1, f"{len(ii1) / len(if1):.3f}", len(ii1), len(if1), len(x1))
        print(p2, f"{len(ii2) / len(if2):.3f}", len(ii2), len(if2), len(x2[0]))
    # assert False
    np.save(os.path.join(OUTDIR, f"{p1}_contacts.{p2}.npy"), ii1)
    np.save(os.path.join(OUTDIR, f"{p2}_contacts.{p1}.npy"), ii2)
    # ---- Compute connectivity matrix
    # assert False
    ii1, rho1, li1 = get_reordered_subset(ii1, rho1, li1)
    ii2, rho2, li2 = get_reordered_subset(ii2, rho2, li2)
    within1 = get_within(rho1, li1, neighbor_dist)
    within2 = get_within(rho2, li2, neighbor_dist)
    np.save(os.path.join(OUTDIR, f"{p1}_within.{p2}.npy"), within1)
    np.save(os.path.join(OUTDIR, f"{p2}_within.{p1}.npy"), within2)

    return


# ----- OX40 specific funs
def get_cdr_residues(x, OUTDIR):
    cdr_residues = {}
    for chain, cdrs in zip(["L", "H"], [('CDRL1', 'CDRL2', 'CDRL3'), ('CDRH1', 'CDRH2', 'CDRH3')]):
        tmp = []
        for cdr in cdrs:
            st, e = x[cdr].split("-")
            st = int(st[1:])
            e = int(e[1:]) + 1
            tmp.append(np.arange(st, e, 1, dtype=int))
        cdr_residues[chain] = np.concatenate(tmp)

    pdb = x["Ag"] + ".pdb"
    fname = os.path.join(OUTDIR, pdb)
    parser = PDBParser()
    structure = parser.get_structure(pdb, fname)[0]

    xs = []
    for atom in structure.get_atoms():
        _, _, chain, (_, rid, _), _ = atom.get_full_id()

        if "261421" in pdb:
            if chain == "A":
                chain = "H"
            elif chain == "B":
                chain = "L"
        else:
            if chain == "A":
                chain = "L"
            elif chain == "B":
                chain = "H"

        if rid in cdr_residues[chain]:
            xyzp = atom.get_coord().reshape((1, 3))
            xs.append(xyzp)
    xs = np.concatenate(xs, axis=0)

    return xs


def compute_aux_vars_CDR(p1, OUTDIR, expand_radius, neighbor_dist,
                         contact_thres1, contact_thres2, device, x2, smooth=True):
    data1 = np.load(os.path.join(OUTDIR, f"{p1}_surface.npz"))
    x1 = data1["pos"]
    li1 = data1["list_indices"]
    rho1 = data1["rho"]
    n1 = data1["normals"]
    # ---- Create smoothed normals
    if smooth:
        fname1 = os.path.join(OUTDIR, f"{p1}_smoothed_normals.npy")
        n1_smoothed = smooth_normals(n1, rho1, li1)
        print(np.sum(n1 * n1_smoothed) / len(n1))
        np.save(fname1, n1_smoothed)
    # ---- calc dist
    x1 = torch.tensor(x1.reshape(len(x1), 1, 3)).to(device)
    x2 = torch.tensor(x2.reshape(1, len(x2), 3)).to(device)
    d = torch.norm(x1-x2, dim=2).cpu().numpy()
    ii1, ii2 = np.where(d < contact_thres1)

    # ---- expand
    rho1_ = rho1[ii1]
    ii1 = np.unique(li1[ii1][(rho1_ > 0) & (rho1_ < expand_radius)])
    # assert False
    np.save(os.path.join(OUTDIR, f"{p1}_contacts.{p1}.npy"), ii1)
    # ---- Compute connectivity matrix
    # assert False
    ii1, rho1, li1 = get_reordered_subset(ii1, rho1, li1)
    within1 = get_within(rho1, li1, neighbor_dist)
    np.save(os.path.join(OUTDIR, f"{p1}_within.{p1}.npy"), within1)

    return

# ----- OX40 specific funs -- end
