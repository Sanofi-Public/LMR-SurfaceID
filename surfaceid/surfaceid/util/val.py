from surface_id.util.utils import (RLIM, SIG, SIG_NORMAL, Aligner1, get_fs, Mol_slim, 
    find_clusters, get_whole_molecule_desc, get_fs_npz, get_xyz, get_centroid,
    get_score1, get_score2, get_normal_npz, get_xyz_npz, transform_library, transform,
    get_batch, Model, compute_aux_vars, gen_descriptors_contact, get_desc_aux,
    search, get_within, smooth_normals, get_reordered_subset)
from Bio.PDB import PDBParser, rotaxis2m, PDBIO
import seaborn as sns
from time import time
import os
import subprocess
import numpy as np
import pandas as pd
import sys
from functools import partial
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from plyfile import PlyData, PlyElement
from copy import deepcopy
import shutil
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
torch.set_num_threads(cpu_count())


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
        _, _, chain, (_, rid, _), _ =  atom.get_full_id()

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