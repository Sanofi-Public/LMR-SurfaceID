import seaborn as sns
from time import time
import os
# import subprocess
import numpy as np
import pandas as pd
from functools import partial
# import tempfile
from multiprocessing import Pool, cpu_count
from Bio.PDB import PDBParser, rotaxis2m, PDBIO
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch_scatter
import torch.optim as optim
import sys
from plyfile import PlyData, PlyElement
from copy import deepcopy
from scipy.special import softmax as scipy_softmax
from sklearn.decomposition import PCA

from surfaceid.surfaceid.model.model import Model, clean_tensor, Mol, DNEG, DPOS, OUTDIR
from surfaceid.surfaceid.util.mol import Mol_with_epitopes

# Globals
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
    """Obtain Surface ID descriptors for the surface patches at the contact reagions
        as defined in the "{p1}_contacts.{p2}.npy". Otherwise it uses entire protein Surface 

    :param p1s: ID for protein 1: Format: PDBID_chain
    :type p1s: str
    :param p2s: ID for protein 2: Format: PDBID_chain
    :type p2s: str
    :param model:  model class used to obtain descriptors for a given protein   
    :type model: Model
    :param device: CPU or GPU
    :type device: Torch.device
    :param rho_max: max value for the radial distance from the patch center 
    :type rho_max: float
    :param outdir: directory where Surface ID descriptors are saved 
    :type outdir: str
    :param fname: npz file containing descriptors for all proteins
    :type fname: str
    :return: pdb,Surface ID descriptors, and patch/vertex indices  
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
    """ helper function for patch alignment where the hit patch aligned to the query patch using the align_model

    :param x: X coordinates for the vertex points on the patch
    :type x: np.array
    :param y: Y coordinates for the vertex points on the patch
    :type y: np.array
    :param z: Z coordinates for the vertex points on the patch
    :type z: np.array
    :param xyz0_mean: center of the query patch
    :type xyz0_mean: np.array
    :param xyz_mean: center of the candidate patch
    :type xyz_mean: np.array
    :param align_model: the model containing transformation parameters to align candidate patch to the query
    :type align_model: Class
    :param idx: Sorted indeces of the loss model
    :type idx: int
    :param n: patch normal vector , defaults to None
    :type n: np.array, optional
    :return: aligned patch and its normal
    """
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


def get_score2_helper(xyz0, fs_raw0, n0, xyz, fs_raw, n, sig, sig_normal=SIG_NORMAL):
    """helper function for computing the alignment score 2.
       this score uses the real space/raw features for the vertexes that are in contact 
       (RLIM=1.5\AA). It computes the feature distance for contact verteces and weighs
       it using the softmax of a Gaussian wieghed V-V distance. It also computers the 
        dot product of the normals for the contact vertices

    :param xyz0: coordinates of the query patch
    :type xyz0: np.array
    :param fs_raw0: raw features for the query from MaSIF preprocessing 
    :type fs_raw0: np.array
    :param n0: normal vectors for the query pacth
    :type n0: np.array
    :param xyz: coordinates of the candidate patch
    :type xyz: np.array
    :param fs_raw: raw features for the candidate from MaSIF preprocessing
    :type fs_raw: np.array
    :param n: normal vectors for the candidate pacth
    :type n: np.array
    :param sig: the spread for the Gaussian wieghts
    :type sig: float
    :param sig_normal: , defaults to SIG_NORMAL
    :type sig_normal: float, optional
    :return: alignment nontacts, score (feature), alignment score(normals)
    """
    r = get_r(xyz0, xyz)  # [N0, N]
    mask = (r < RLIM).astype(np.float32)
    r = 100. * (1. - mask) + r * mask
    nactive = np.sum(mask.sum(axis=0) > 0)
    w = scipy_softmax(-r**2 / sig**2, axis=0)
    f_matrix = np.sum(
        np.square(fs_raw0[:, None, :] - fs_raw[None, :, :]), axis=-1)
    score = (w * mask * f_matrix).sum() / nactive
    n_matrix = np.sum(n0[:, None, :] * n[None, :, :], axis=-1)
    n_score = (w * mask * n_matrix).sum() / \
        nactive  # normals are already smoothed
    return nactive, score, n_score


def get_score2(xyz0, fs_raw0, n0, xyz, fs_raw, n, sig):
    """ averages the alignment score2 permutating candidate and hit patches

    :param xyz0: coordinates of the query patch
    :type xyz0: np.array
    :param fs_raw0: MaSIF raw feature for the query
    :type fs_raw0: np.array
    :param n0: normal vectors on the query
    :type n0: np.array
    :param xyz: coordinates of the candidate patch
    :type xyz: np.array
    :param fs_raw: MaSIF raw feature for the query
    :type fs_raw: np.array
    :param n: normal vectors on the candidate
    :type n: np.array
    :param sig: gaussian spread parameter
    :type sig: float
    :return: ncontacts, mean scores using features and normals
    """
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
    """ computes PCA

    :param xyz: xyz
    :type xyz: np.array
    """
    pca = PCA(n_components=3)
    pca.fit(xyz)
    q1, q2, q3 = pca.components_
    q3 = np.sign(np.dot(np.cross(q1, q2), q3)) * q3
    return q3


def get_fs(query, pdbs_desc, x_desc, idxs_desc):
    """returns the SurfaceID descriptors
    :param query: the protein id 
    :type query: str
    :param pdbs_desc: protein ids
    :type pdbs_desc: np.array
    :param x_desc: surface id descriptors 
    :type x_desc: np.array
    :param idxs_desc: surface patch indices
    :type idxs_desc: np.array
    :return: surface id descriptors and patch indeces
    """
    iselect = pdbs_desc == query

    return x_desc[iselect], idxs_desc[iselect]


def get_xyz_npz(fname, idxs=None):
    """returns x coordinates of the 
        vertex points on the protein surface 

    :param fname: npz file name
    :type fname: str
    :param idxs: vertex id, defaults to None
    :type idxs: int, optional
    :return: x coordinates
    """
    data = np.load(fname)
    vs = data["pos"]
    return vs[idxs]


def get_normal_npz(fname, idxs=None):
    """ normal vectors on protein surface

    :param fname: npz file name 
    :type fname: str
    :param idxs: vertex id, defaults to None
    :type idxs: int , optional
    :return: normal vectors
    """
    data = np.load(fname)
    vs = data["normals"]
    return vs[idxs]


def get_xyz(plydata, idxs=None):
    """ returns the xyz coordinates 
        on portein surface

    :param plydata: plydata object
    :type plydata: plydata 
    :param idxs: vertex id , defaults to None
    :type idxs: int, optional
    :return: xyz coordinates
    :rtype: _type_
    """
    vs = plydata["vertex"]
    x, y, z = vs["x"], vs["y"], vs["z"]
    if idxs is not None:
        return np.stack([x, y, z]).T[idxs]
    else:
        return x, y, z


def get_fs_npz(fname, idxs):
    """ returns surface features on 
        protein surface

    :param fname: npz file name
    :type fname: str
    :param idxs: vertex/patch ids
    :type idxs: np.array
    :return: MaSIF features
    """
    x = np.load(fname)["x_initial"][idxs]
    x = np.maximum(x, -1.0)
    x = np.minimum(x, 1.0)

    return x


def get_centroid(xyz):
    return np.mean(xyz, axis=0)


def transform(x, y, z, a, b, g, Tx, Ty, Tz, xyz0_mean, xyz_mean):
    """ apply 3D rotation/translation 
    :param x: X coordinates
    : type x: np.array
    :param y: Y coordinates
    : type y: np.array
    :param z: Z coordinates
    : type z: np.array
    :param a: alpha rotation angle
    : type a: float
    :param b: beta rotation angle
    : type b: float
    :param c: gamma rotation angle
    : type c: float
    :param Tx: translation in X direction
    : type Tx: float
    :param Ty: translation in y direction
    : type Ty: float
    :param Tz: translation in Z direction
    : type Tz: float
    :param xyz0_mean: query patch center
    : type xyz0_mean: float
    :param xyz_mean: candidate patch center
    : type xyz_mean: float
    :return: transformed coordinates
    """
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
    """ rotates the coordinates back

    :param x: X coordinates
    :type x: np.array
    :param y: Y coordinates
    :type y: np.array
    :param z: Z coordinates
    :type z: np.array
    :param a: alpha rotation angle
    :type a: float
    :param b: beta rotation angle
    :type b: float
    :param g: gamma rotation angle
    :type g: float
    :return: transformed oordinates
    """
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
    """ averages over the 
        patch normal vectors

    :param n: patch normal vectors
    :type n: np.array
    :return: averaged normal vectors
    """
    n = np.mean(n, axis=0)
    return n / np.sqrt(np.sum(n**2))


def get_desc_aux(pdbs):
    """ returns the indices where a 
        given protein id starts and ends
        in Surface ID descriptor file 
    :param pdbs: pdb field in the descriptor file
    :type pdbs: np.array
    :return: number of patches corresponding to each 
            protein in the area of interest,
            unique protein ids, and the start-end indices
            for each protein in the descriptor file
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


def search(o1, 
           rho, 
           list_indices, 
           within, 
           library,
           library_within, 
           pdbs_unique, 
           patch_sizes,
           pdbs, 
           x_desc, 
           idxs_desc, 
           mini_bs, 
           device, 
           nmin_pts, 
           expand_radius, 
           nmin_pts_library,
           thres, 
           target, 
           idxs_contact):
    """ performs the search for all pairs between the query and candidate
        first, it computes the descriptor distance between all pair of verticies
        between  query and candidate. it selects pairs within the descriptor 
        distance thresheold. if number of hits between query and candidate exceeds
        the threshold arameter, hit grouping is performed to unify vertex hits with
        their nearest neighbors and form a hit area on each protein surface.

    :param o1: Surface ID descriptors for the query protein
    :type o1: torch.tensor
    :param rho: radial distribution of vertices on a patch from the patch center
    :type rho: np.array
    :param list_indices: embedded list containing the vertex ids within each patch
    :type list_indices: np.array
    :param within: vertex ids that fall within the neighbor_dist for the query patch
    :type within: np.array
    :param library: protein ids in the library
    :type library: np.array
    :param library_within: embedded list of vertex ids corresponding to the vertices
                           within the neighbor_dist of each patch in the candidate protein
    :type library_within: np.array
    :param pdbs_unique: unique pdb ids
    :type pdbs_unique: np.array
    :param patch_sizes: number of patches for every candidate protein within the area of interest
    :type patch_sizes: np.array
    :param pdbs: protein names corresponding to each patch in the library following the Surface ID descriptor
    :type pdbs: np.array
    :param x_desc: Surface ID descriptors
    :type x_desc: torch.tensor
    :param idxs_desc: patch ids for the corresponding descriptors
    :type idxs_desc: torch.tensor
    :param mini_bs: mini batches to perform the search
    :type mini_bs: torch.tensor
    :param device: CPU or GPU
    :type device: str
    :param nmin_pts: minimum number of hit patches on query for identifying it as a hit  
    :type nmin_pts: int
    :param expand_radius: the expansion radius around the hit vertices to define the hit area following the clustering step
    :type expand_radius: float
    :param nmin_pts_library: minimum number of hit patches on candidate protein for identifying it as a hit
    :type nmin_pts_library: int
    :param thres: threshld used for descriptor distance to identify two patches as similar
    :type thres: float
    :param target: protein id for the query 
    :type target: str
    :param idxs_contact: patch indices for located at the area of contact 
    :type idxs_contact: np.array
    :return: 
    """
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

def find_clusters(hits, within, rho, list_indices, nmin_pts=200, expand_radius=3.0):
    """ unites the hit verteces using nearest neighbors within the expansion radius.
        the clusters that satisfy the cluster size threshold are returned as hit area.

    :param hits: hit vertex points/patches that satisfy the descriptor distance cutoff 
    :type hits: np.array
    :param within: 
    :type within: np.array
    :param rho: radial distances of vertecies wrt patch center
    :type rho: np.array
    :param list_indices: embedded list of vertex ids for each patch
    :type list_indices: list
    :param nmin_pts: cutoff for cluster size, defaults to 200
    :type nmin_pts: int, optional
    :param expand_radius: distance to include the nearest neighbors, defaults to 3.0
    :type expand_radius: float, optional
    :return: expanded hit area (vertex points)
    :rtype: _type_
    """
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
    """ returns the Surface ID descriptors for the whole entire protein
    :param target: protein name
    :type target: str
    :param model: Surface ID model 
    :type model: Model
    :param rho_max: max radial cutoff for patch vertices  
    :type rho_max: float
    :param bs:batch size, defaults to 256
    :type bs: int, optional
    :param device: cpu or gpu, defaults to "cpu"
    :type device: str, optional
    :param outdir: directory where descriptors are saved, defaults to None
    :type outdir: int, optional
    :return: descriptors, protein name
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




def get_within(rho, list_indices, neighbor_dist):
    """ returns the vertex points within neighbor_dist or patch center

    :param rho: radial distances of vertex points wrt patch center
    :type rho: np.array
    :param list_indices: embedded list of vertices within each patch
    :type list_indices: list
    :param neighbor_dist: distance cutoff
    :type neighbor_dist: float
    :return: vetex points within the cutoff distance from each patch
    :rtype: _type_
    """
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
    """ returns re-ordered subset of points 

    :param idxs:  subset vertices
    :type idxs: np.array
    :param rho: radial distances of vertex points wrt patch center
    :type rho: np.array
    :param list_indices: embedded list of vertices within each patch
    :type list_indices: list
    :return: re-ordered subset of points
    :rtype: _type_
    """
    idxs_new = np.arange(len(idxs), dtype=np.int32)
    reorder_map = np.full(np.max(list_indices)+1, -1, dtype=np.int32)
    reorder_map[idxs] = idxs_new
    rho_subset, list_indices_subset = rho[idxs], reorder_map[list_indices[idxs]]

    return idxs_new, rho_subset, list_indices_subset


def smooth_normals(n, rho, list_indices, sig_normal=SIG_NORMAL):
    """ returns the mean of normal vetors for each patch using the 
        vertex points within a cutoff distance of 3*SIG_NORMA. 

    :param n: normal vectors for each vertex point
    :type n: np.array
    :param rho: radial distances of vertex points wrt patch center
    :type rho: np.array
    :param list_indices: embedded list of vertices within each patch
    :type list_indices: list
    :param sig_normal: spread , defaults to SIG_NORMAL
    :type sig_normal: int, optional
    :return: mean normal
    :rtype: _type_
    """
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
    #""" p1, p2 are binding partners. Compute smoothed normals,
    #contact region indices, and a matrix that tells which points are connected
    #"""
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

    return None


# ----- OX40 specific funs
def get_cdr_residues(x, OUTDIR):
    """extract the coordinates of CDR residues from the x DataFrame 

    :param x: pd.DataFrame containing the CDRs
    :type x: pd.DataFrame
    :param OUTDIR: directory containing the PDB files
    :type OUTDIR: str
    :return:  XYZ coordinates of the CDR residues
    :rtype: _type_
    """
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


def compute_aux_vars_CDR(p1, 
                         OUTDIR, 
                         expand_radius, 
                         neighbor_dist,
                         contact_thres1, 
                         contact_thres2, 
                         device, 
                         x2, 
                         smooth=True):
    """ idntifies and writes the surface patches at the interace, the mean normal vectors,
        and vertex points that are within a utoff distance from each patch at the interface
    :paramp1: protein name
    :type p2: str
    :OUTDIR: the directory where the MaSIF surface files are saved
    :type OUTDIR: str
    :param expand_radius: cutoff distace to obtain the nearest eighbors
    :type expand_radius: float
    :param neighbor_dist: list of vertex points within a patch
    :type neighbor_dist: list
    :param contact_thres1: cutoff distance for protein 1 to identify the patches at the area of interest 
    :type contact_thres1: float
    :param contact_thres2: cutoff distance for protein 2 to identify the patches at the area of interest
    :type contact_thres2: float
    :param x2: coordinates of verecies for protein 2
    :type x2: np.array
    :return: 
    :rtype: _type_
    """
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
    np.save(os.path.join(OUTDIR, f"{p1}_contacts.{p1}.npy"), ii1)
    # ---- Compute connectivity matrix
    # assert False
    ii1, rho1, li1 = get_reordered_subset(ii1, rho1, li1)
    within1 = get_within(rho1, li1, neighbor_dist)
    np.save(os.path.join(OUTDIR, f"{p1}_within.{p1}.npy"), within1)

    return None
