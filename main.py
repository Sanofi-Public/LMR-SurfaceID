from surfaceid.util.utils import (SEARCH, ALIGN, DESC, RLIM, SIG,  Aligner1, get_fs,
                                  get_whole_molecule_desc, get_fs_npz, get_xyz, get_centroid,
                                  get_score1, get_score2, get_normal_npz, get_xyz_npz, transform_library,
                                   Model, compute_aux_vars, gen_descriptors_contact, get_desc_aux,
                                  search, get_within)


import logging
import os
import argparse
import numpy as np
import pandas as pd
import sys
from multiprocessing import cpu_count
import matplotlib.pyplot as plt
from time import time

from Bio.PDB import PDBParser, PDBIO

import torch
import torch.nn as nn
import torch.optim as optim
from plyfile import PlyData, PlyElement
from copy import deepcopy
import shutil

torch.set_num_threads(cpu_count())
CONTACT = False
DESC = False
SEARCH = False
ALIGN = False
SAVEPLY = False
HEATMAP = False
ZIP = False
#
CONTACT = True
DESC = True
SEARCH = True
ALIGN = True
SAVEPLY = True
ZIP = True
#
REMOVE_OLD = ALIGN and SAVEPLY and SEARCH
CATALOG_PAIRS = "data/20201107_SAbDab_catalog_pairs.tsv"

expand_radius = 2.0
neighbor_dist = 3.0
nmin_pts = 40  # for grouping target decoys
nmin_pts_library = 30  # library epitope
# Model 
prefix = "final_002"
device = 'cpu'

# ---- Identify antigens
case = "4fqi_AB"
OUTDIR_RESULTS = f"{case}_results"
OUTDIR = "data/20201107_SAbDab_masif"
target = "4fqi_AB"

contact_thres1 = 3.0  # for identifying iface points
contact_thres2 = None  # for distance based interacting points
contact_mode = "dist"

npatience = 100

# --- 6AA
params = {"rho_max": 6.0, "nbins_rho": 5, "nbins_theta": 16, "num_in": 5,
            "neg_margin": 10, "add_center_pixel": True,
            "share_soft_grid_across_channel": True,
            "conv_type": "sep", "num_filters": 512, "weight_decay": 1e-2,
            "dropout": 0.1, "min_sig": 5e-2, "lr": 5e-4}
thres = 4.0

savedir = "models"

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger("🧪 SurfaceID")

parser = argparse.ArgumentParser(
    prog="python main.py ...",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description=(
        """\
                
                -----------------------------------------------------
                Surface Similarity Search via Geometric Deep Learning
                -----------------------------------------------------
        """
    ),
)
parser.add_argument(
    "--pdb", default="4fqi_AB", help="Input file for execution"
)
parser.add_argument(
    "--device", default="cpu", help="Device to compute"
)
parser.add_argument(
    "--params", default="config/params.yml", help="Number of cores"
)
parser.add_argument(
    "--device", default="3", help="Number of cores"
)
# parse the arguments
(args) = parser.parse_args()

if torch.cuda.is_available():
    device = torch.cuda.current_device()

if SEARCH or DESC or ALIGN:
    path = os.path.join(savedir, f"model_{prefix}.pth")
    model = Model(**params)
    model = model.to(device)
    model.dthetas = model.dthetas.to(device)
    model.load_state_dict(torch.load(
        path, map_location=torch.device('cpu')))
    model.eval()

df = pd.read_csv(CATALOG_PAIRS, sep="\t")
df.Ab = df.apply(lambda x: f'{x.id}_{x.Ab}', axis=1)
df.Ag = df.apply(lambda x: f'{x.id}_{x.Ag}', axis=1)
logger.info(f"# of pairs: {len(df)}")
chains = set(df.Ab.tolist() + df.Ag.tolist())
logger.info(f"# of chains: {len(chains)}")
chains_processed = [chain for chain in chains if os.path.exists(
    f"{OUTDIR}/{chain}_surface.npz")]
logger.info(
    f"# of chains processed: {len(chains_processed)}")
df = df[df.apply(lambda x: (x["Ag"] in chains_processed) and (
    x.Ab in chains_processed), axis=1)].reset_index(drop=True)
logger.info(f"# of pairs after filter: {len(df)}")

if os.path.exists(OUTDIR_RESULTS) and REMOVE_OLD:
    shutil.rmtree(OUTDIR_RESULTS)
os.makedirs(OUTDIR_RESULTS, exist_ok=True)

if CONTACT:
    for i, x in df.iterrows():
        p1 = x.Ab
        p2 = x.Ag
        logger.info(f"{i}, {p1}, {p2}")
        compute_aux_vars(p1, p2, OUTDIR, expand_radius, neighbor_dist,
                            contact_mode, contact_thres1, contact_thres2, device, smooth=False)

if DESC:
    p1s = df.Ab.to_numpy()
    p2s = df.Ag.to_numpy()
    fname = f"nn_desc.{case}.npz"
    pdbs, x_desc, idxs_desc = gen_descriptors_contact(
        p1s, p2s, model, device, params["rho_max"], OUTDIR, fname)
    patch_sizes, pdbs_unique, starts, ends = get_desc_aux(pdbs)

# ---------------- All vs all search
# Load descriptors
# For each Ag epitope, compare against all other
if not DESC and (ALIGN or SEARCH):
    fname = f"nn_desc.{case}.npz"
    data = np.load(fname)
    pdbs = data.pdbs
    x_desc = data.x
    idxs_desc = data["idxs"]
    patch_sizes, pdbs_unique, starts, ends = get_desc_aux(pdbs)

if SEARCH:

    # libraries
    library = df.apply(lambda x: f"{x['Ag']}.{x['Ab']}", axis=1).to_list()
    library_within = [np.load(os.path.join(OUTDIR, x)) for x in df.apply(
        lambda x: f"{x['Ag']}_within.{x['Ab']}.npy", axis=1).to_list()]
    DIM = params["num_filters"]
    n = len(pdbs_unique)
    summary = []
    target = "4fqi_AB"
    st_target = time()
    o1, p1 = get_whole_molecule_desc(
        target, model, params["rho_max"], bs=128, device=device, outdir=OUTDIR)
    o1 = torch.tensor(o1).to(device).view(len(o1), 1, DIM)
    mini_bs = int(64 * 1000 / len(o1))
    rho = p1.rho.cpu().numpy()
    list_indices = p1.list_indices.cpu().numpy()

    # Get inclusion matrix
    within = get_within(rho, list_indices, neighbor_dist)
    summary.extend(search(o1, rho, list_indices, within, library, library_within, pdbs_unique, patch_sizes,
                            pdbs, x_desc, idxs_desc, mini_bs, device, nmin_pts, expand_radius, nmin_pts_library,
                            thres, target, idxs_contact=None))
    # remove o1
    o1 = o1.cpu()
    del o1
    (f"{target} / dt = {time() - st_target:.3f}")

    df_hits = pd.DataFrame(summary, columns=["target", "library epitope", "target_nhits", "target_nexpanded", "mean_desc_dist",
                            "library_nhits", "library_nexpanded", "frac_library_hits", "target_expanded_indices", "library_epitope_indices"])
    df_hits["frac_expanded"] = df_hits["target_nhits"] / \
        df_hits["target_nexpanded"]
    df_hits["frac_geo"] = np.sqrt(
        df_hits["frac_library_hits"] * df_hits["frac_expanded"])
    df_hits["target_expanded_indices"] = df_hits["target_expanded_indices"].apply(
        lambda x: list(x))
    df_hits["library_epitope_indices"] = df_hits["library_epitope_indices"].apply(
        lambda x: list(x))
    df_hits.to_csv(os.path.join(
        OUTDIR_RESULTS, f"{case}_top_hits.tsv"), sep="\t", index=False)

if ALIGN:
    df_hits = pd.read_csv(os.path.join(
        OUTDIR_RESULTS, f"{case}_top_hits.tsv"), sep="\t")
    df_hits["target_expanded_indices"] = df_hits["target_expanded_indices"].apply(
        lambda x: np.asarray(eval(x), dtype=np.int32))
    df_hits["library_epitope_indices"] = df_hits["library_epitope_indices"].apply(
        lambda x: np.asarray(eval(x), dtype=np.int32))
    df_hits["id"] = -1
    npatience = 100
    DIM = params["num_filters"]
    librarys = df.apply(lambda x: f'{x["Ag"]}.{x["Ab"]}', axis=1).tolist()
    target = "4fqi_AB"
    id_counter = 0
    st_target = time()
    logger.info(f"/----{target}")
    if SAVEPLY:
        ply_target = deepcopy(PlyData.read(
            os.path.join(OUTDIR, f"{target}.ply")))
    o_target, p_target = get_whole_molecule_desc(
        target, model, params["rho_max"], bs=128, device=device, outdir=OUTDIR)
    mini_bs = int(64 * 1000 / len(o_target))
    for library in librarys:
        df_ = df_hits[(df_hits["target"] == target) & (
            df_hits["library epitope"] == library)]
        nsample = min(10, len(df_))
        df_ = df_.sort_values("frac_geo", ascending=False).head(nsample)
        logger.info(f"/--{library}")
        for ie, entry in df_.iterrows():
            st = time()
            library_p1, library_p2 = entry.library.epitope.split(".")
            # assert False
            # Load library data
            fname = os.path.join(OUTDIR, f"{library_p1}_surface.npz")
            fs, idxs_library0 = get_fs(
                entry["library epitope"], pdbs, x_desc, idxs_desc)
            # Get a subselection of match points
            ii = entry["library_epitope_indices"]
            idxs_library = idxs_library0[ii]
            fs = fs[ii]
            fs_raw = get_fs_npz(fname, idxs_library)
            xyz = get_xyz_npz(fname, idxs_library)
            n = get_normal_npz(fname, idxs_library)
            x, y, z = deepcopy(xyz).T
            xyz_mean = get_centroid(xyz)
            xyz -= xyz_mean

            fname_target = os.path.join(OUTDIR, f"{target}_surface.npz")
            idxs_target = entry["target_expanded_indices"]
            
            if (len(idxs_target) < nmin_pts) or (len(idxs_target) > 5000):
                continue
            fs0 = o_target[idxs_target]
            fs_raw0 = get_fs_npz(fname_target, idxs_target)
            xyz0 = get_xyz_npz(fname_target, idxs_target)
            x0, y0, z0 = xyz0.T
            n0 = get_normal_npz(fname_target, idxs_target)
            xyz0_mean = get_centroid(xyz0)
            xyz0 -= xyz0_mean

            # -- compute hits
            with torch.set_grad_enabled(False):
                fs0 = torch.tensor(fs0).to(device).view(len(fs0), 1, DIM)
                n2 = len(fs)
                ds = []
                for i1 in range(0, n2, mini_bs):
                    i2 = min(i1+mini_bs, n2)
                    fs_ = torch.tensor(fs[i1:i2]).to(
                        device).view(1, i2-i1, DIM)
                    d = torch.norm(fs0 - fs_, dim=2)
                    ds.append(d)
                d = torch.cat(ds, dim=1)
                mask = (d < thres).float()
            T_zero = np.asarray([0, 0, 0])

            # -- 1st alignment
            # if too big, gpu memory run out for alignment
            if len(idxs_target) > 2500:
                nbin = 2
            elif len(idxs_target) > 1000:
                nbin = 3
            else:
                nbin = 4

            npairs = mask.sum().item()
            aligner1 = Aligner1(xyz0, xyz, mask.clone().detach(
            ), nbin, device, npairs, float_type=np.float32)
            aligner1 = aligner1.to(device)
            optimizer = optim.Adam(aligner1.parameters(), lr=5e-2)
            nbad = 0
            min_loss = np.infty
            i = 0
            while True:
                i += 1
                optimizer.zero_grad()
                loss, losses = aligner1()
                loss.backward()
                optimizer.step()
                losses = losses.detach().cpu().numpy()
                idx1 = np.argmin(losses).item()
                loss = losses[idx1]
                if loss < min_loss * 0.999:
                    nbad = 0
                    min_loss = loss
                else:
                    nbad += 1
                if (nbad == npatience):
                    print(f"1 / {i} loss_min", losses.min().item(), idx1)
                    break
                if i == 0:
                    print(f"1 / {i} loss_min", losses.min().item(), idx1)
            xyz_patch, n = transform_library(
                x, y, z, T_zero, xyz_mean, aligner1, idx1, n)
            # collect stats
            df_hits.loc[ie, f"# hits repeat"] = (
                mask.sum(1) > 0).sum().item()
            df_hits.loc[ie, f"# pairs"] = npairs
            mask = mask.cpu().numpy()
            df_hits.loc[ie, f"aln1_score1"] = get_score1(
                xyz0, xyz_patch, mask, npairs)  # losses.min().item()
            try:
                ncontact, n_score, score2 = get_score2(
                    xyz0, fs_raw0, n0, xyz_patch, fs_raw, n, SIG)
                fs0 = o_target[idxs_target]
                _, _, score3 = get_score2(
                    xyz0, fs0, n0, xyz_patch, fs, n, SIG)
            except:
                n_score, score2, score3 = 0., 100., 1000.
            df_hits.loc[ie, f"aln1_score2"] = score2
            df_hits.loc[ie, f"aln1_score3"] = score3
            df_hits.loc[ie, f"aln1_normal_score"] = n_score
            df_hits.loc[ie, f"aln1_ncontact"] = ncontact
            id_counter += 1
            df_hits.loc[ie, "id"] = id_counter

            logger.info(f"Time for {library}: {time()-st:.2f}")

            # and ((df_hits.loc[ie, f"aln1_score2"] < 0.9) or (df_hits.loc[ie, f"frac_library_hits"] > 0.3)):
            if SAVEPLY:
                fname_library = os.path.join(OUTDIR, f"{library_p1}.ply")
                if not os.path.exists(fname_library):
                    continue

                # -- create ply after aln1
                # transform contact patch
                fname = os.path.join(OUTDIR, f"{library_p1}_surface.npz")
                els = []
                for tag, iis in zip(["hit_patch", "contact_patch"], [idxs_library, idxs_library0]):
                    xyzp = get_xyz_npz(fname, iis)
                    xp, yp, zp = xyzp.T
                    xyz_patch_, _ = transform_library(
                        xp, yp, zp, xyz0_mean, xyz_mean, aligner1, idx1, None)
                    vertex_patch = np.asarray([tuple(q) for q in xyz_patch_], dtype=[
                                                ('px', 'f4'), ('py', 'f4'), ('pz', 'f4')])
                    el = PlyElement.describe(vertex_patch, tag)
                    els.append(el)
                # transform whole molecule
                ply_library = deepcopy(PlyData.read(fname_library))
                xp, yp, zp = get_xyz(ply_library, idxs=None)
                xyzp, _ = transform_library(
                    xp, yp, zp, xyz0_mean, xyz_mean, aligner1, idx1, n=None)
                # Alter vertices and save
                fname = os.path.join(
                    OUTDIR_RESULTS, f"{id_counter}_{entry['library epitope']}_{target}.ply")
                ply_library["vertex"]["x"] = xyzp[:, 0]
                ply_library["vertex"]["y"] = xyzp[:, 1]
                ply_library["vertex"]["z"] = xyzp[:, 2]
                ply_library.elements = ply_library.elements + tuple(els)
                ply_library.write(fname)

                # ---- Add patch info to target
                xyz0 = get_xyz_npz(fname_target, idxs_target)
                x0, y0, z0 = xyz0.T
                vs = np.asarray(list(zip(x0, y0, z0)), dtype=[
                                ('px', 'f4'), ('py', 'f4'), ('pz', 'f4')])
                el = PlyElement.describe(
                    vs, f"patch_{id_counter}_{entry['library epitope']}")
                ply_target.elements = ply_target.elements + (el,)

                # ---- Alter and save pdb
                # Parse the structure file
                fname = os.path.join(OUTDIR, f"{library_p1}.pdb")
                if not os.path.exists(fname):
                    continue
                parser = PDBParser()
                structure = parser.get_structure(library, fname)[0]

                # Iterate through all atoms and transform
                for atom in structure.get_atoms():
                    xyzp = atom.get_coord().reshape((1, 3))
                    xp, yp, zp = xyzp.T
                    xyzp, _ = transform_library(
                        xp, yp, zp, xyz0_mean, xyz_mean, aligner1, idx1, n=None)
                    atom.set_coord(xyzp[0])
                fname = os.path.join(
                    OUTDIR_RESULTS, f"{id_counter}_{entry['library epitope']}_{target}.pdb")
                io = PDBIO()
                io.set_structure(structure)
                io.save(fname)
    logger.info(f"Time for {target}: {time()-st_target:.2f}")
    if SAVEPLY:
        out_target_fname = os.path.join(
            OUTDIR_RESULTS, f"{target}_ref.ply")
        ply_target.write(out_target_fname)
    df_hits.to_csv(os.path.join(
        OUTDIR_RESULTS, f"{case}_top_hits_aligned.tsv"), sep="\t", index=False)

    # ---- Ranking vs. scores
    df_ = df_hits[(~df_hits["aln1_score2"].isna())]
    for rank in ["mean_desc_dist", "frac_expanded", "frac_library_hits", "frac_geo"]:
        plt.close()
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        ax.scatter(df_[rank], df_["aln1_score2"], label="aln1")
        ax.set_ylim(
            [0, max(np.percentile(df_["aln1_score2"], 90) * 1.1, 1.0)])
        ax.set_xlabel(rank)
        ax.set_ylabel(
            f"mean feat dist of closest vertex pairs ({RLIM}) cutoff")
        ax.legend(loc="upper right")
        plt.savefig(os.path.join(
            OUTDIR_RESULTS, f"{case}_{rank}_vs_score2.png"), dpi=200, bbox_inches="tight")
        plt.close()
        ascending = True if rank == "mean_desc_dist" else False
        df__ = df_.sort_values(
            rank, ascending=ascending).reset_index(drop=True)[:50]
        iselect = df__["aln1_score2"] < 0.9
        nhit = np.sum(iselect)
        logger.info(rank, nhit)

    # change var name
    df_hits = pd.read_csv(os.path.join(
        OUTDIR_RESULTS, f"{case}_top_hits_aligned.tsv"), sep="\t")
    cols = ["library epitope", "aln1_score2",
            "aln1_score3", "aln1_ncontact", "aln1_normal_score"]
    cols_new = ["library_epitope", "spatial_sim",
                "spatial_sim_desc", "ncontact", "normal_score"]
    df_hits.rename(
        columns={c: cn for c, cn in zip(cols, cols_new)}, inplace=True)
    fname = os.path.join(OUTDIR_RESULTS, f"{case}_top_hits_aligned.tsv")
    df_hits.to_csv(fname, sep="\t", index=False)
    df_annots = pd.read_csv(CATALOG_PAIRS, sep="\t")
    df_annots1 = deepcopy(df_annots)
    df_annots1["id"] = df_annots1["id"] + "_" + df_annots1["Ag"] + \
        "." + df_annots1["id"] + "_" + df_annots1["Ab"]
    df_annots2 = deepcopy(df_annots)
    df_annots2["id"] = df_annots2["id"] + "_" + df_annots2["Ab"] + \
        "." + df_annots2["id"] + "_" + df_annots2["Ag"]
    df_annots = pd.concat([df_annots1, df_annots2]).reset_index(drop=True)
    df_hits = pd.merge(df_hits, df_annots, left_on="library_epitope",
                        right_on="id", how="left", suffixes=("", "_"))
    df_hits.to_csv(fname, sep="\t", index=False)

    df_ = df_hits[(~df_hits["spatial_sim"].isna())].reset_index(drop=True)
    ncontact_thres = 35
    for sim_thres in [0.7]:
        rank = "library_nhits"
        ascending = False
        for mm in [10, 25, 50, 100, -1]:
            df__ = df_.sort_values(
                rank, ascending=ascending).reset_index(drop=True)[:mm]
            ihem = df__["antigen_name"].str.contains("hem") | df__["compound"].str.contains(
                "hem") | df__["antigen_name"].str.contains("haem") | df__["compound"].str.contains("haem")
            iselect1 = (df__["spatial_sim"] < sim_thres) & (
                df__["ncontact"] > ncontact_thres) & ihem
            nhit_ha = np.sum(iselect1)
            iselect = (df__["spatial_sim"] < sim_thres) & (
                df__["ncontact"] > ncontact_thres) & ~ihem
            nhit_nonha = np.sum(iselect)
            logger.info(f"Top {mm}", nhit_ha, nhit_nonha)
            logger.info((df__[iselect][cols_new + [rank, "compound",
                        "antigen_name", "id"]].sort_values("ncontact")))
