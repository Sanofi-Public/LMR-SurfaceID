from src.util.utils import (RLIM, SIG, get_fs,
                                  get_whole_molecule_desc, get_fs_npz, get_xyz, get_centroid,
                                  get_score1, get_score2, get_normal_npz, get_xyz_npz, transform_library,
                                  Model, compute_aux_contact, gen_descriptors_contact, get_desc_aux,
                                            search, compute_aux_coors,get_within,compute_aux_whole)

from src.util.Config import Config
from src.model.aligner import Aligner1

import logging
import os
import argparse
import numpy as np
import pandas as pd
import sys
from multiprocessing import cpu_count
import matplotlib.pyplot as plt
from time import time
import yaml

from Bio.PDB import PDBParser, PDBIO
import torch
import torch.nn as nn
import torch.optim as optim
from plyfile import PlyData, PlyElement
from copy import deepcopy
import shutil


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger("🧪 SurfaceID")

argparser = argparse.ArgumentParser(
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
argparser.add_argument(
    "--device", default="cpu", help="Device to compute",dest='device'
)
argparser.add_argument(
    "--params", default="config/config.yml", help="config file with model parameter"
)
argparser.add_argument(
    "--library_csv", help="search library",dest='lib_csv', default=None
)
# parse the arguments
(args) = argparser.parse_args()
if args.lib_csv: print(args.lib_csv)
config = yaml.safe_load(open(args.params, "r"))
logger.info(f"Parsing config file {args.params}")
device = args.device
conf=Config(args.params)

if args.lib_csv:
    conf.CATALOG_PAIRS = args.lib_csv
    
if device == 'cpu':
    torch.set_num_threads(cpu_count())

elif torch.cuda.is_available():
    device = torch.cuda.current_device()

if conf.SEARCH or conf.DESC or conf.ALIGN:
    path = os.path.join(conf.savedir, f"model_{conf.prefix}.pth")
    model = Model(**conf.params)
    model = model.to(device)
    model.dthetas = model.dthetas.to(device)
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()

df = pd.read_csv(conf.CATALOG_PAIRS, sep=",")
if conf.CONTACT:
    df['Ab'] = df.apply(lambda x: f"{x['id']}_{x['Ab']}", axis=1)
    df['Ag'] = df.apply(lambda x: f"{x['id']}_{x['Ag']}", axis=1)
    mix_region = False
    full_surface = False
elif conf.RESTRICT:
    df['Ab'] = df.apply(lambda x: f"{x['id']}_{x['chain']}", axis=1)
    df['Ag'] = df.apply(lambda x: f"{x['id']}_{x['chain']}", axis=1)
    mix_region = False
    full_surface = False

else:
    #search full surface or lookup df['region']
    df['Ab'] = df.apply(lambda x: f"{x['id']}_{x['chain']}", axis=1)
    df['Ag'] = df.apply(lambda x: f"{x['id']}_{x['chain']}", axis=1)
    if 'region' in df.columns:
        mix_region = True
    
logger.info(f" # of pairs: {len(df)}")
chains = set(df['Ab'].tolist() + df['Ag'].tolist())
logger.info(f" # of chains: {len(chains)}")

chains_processed = [chain for chain in chains if os.path.exists(
    f"{conf.OUTDIR}/{chain}_surface.npz")]

logger.info(f" # of chains processed: {len(chains_processed)}")


df = df[df.apply(lambda x: (x['Ab'] in chains_processed) and (
    x['Ag'] in chains_processed), axis=1)].reset_index(drop=True)

logger.info(f" # of pairs after filter: {len(df)}")

if os.path.exists(conf.OUTDIR_RESULTS) and conf.REMOVE_OLD:
    shutil.rmtree(conf.OUTDIR_RESULTS)
os.makedirs(conf.OUTDIR_RESULTS, exist_ok=True)

if conf.CONTACT:
    for i, x in df.iterrows():
        p1 = x['Ab']
        p2 = x['Ag']
        logger.info(f" Contacts for : {i}, {p1}, {p2}")
        compute_aux_contact(p1,
                         p2, 
                         conf.OUTDIR, 
                         conf.expand_radius, 
                         conf.neighbor_dist,
                         conf.contact_mode, 
                         conf.contact_thres1, 
                         conf.contact_thres2,
                         device, 
                         smooth=False)
elif conf.RESTRICT:
    for i, x in df.iterrows():
        p = x['Ab']
        x = np.genfromtxt(f"{conf.OUTDIR}/{p}.xyz")
        logger.info(f" extracting area of interest for : {i}, {p}")
        compute_aux_coors(p,
                         conf.OUTDIR, 
                         conf.expand_radius, 
                         conf.neighbor_dist,
                         conf.contact_thres1, 
                         conf.contact_thres2,
                             device,x,
                         smooth=False)
elif mix_region:
    # c: contact,contact pairs in consequetive rows, selects upper row for search)
    #R:read xyz
    #F: Full
    Cs = df[df['region']=='C']
    Rs = df[df['region']=='R']
    Fs = df[df['region']=='F']
    if len(Cs) > 0 :
        nCs = Cs.to_numpy()[:,:2]
        nCs = nCs.reshape( (len(nCs)//2,-1)) [:,[0,1,3]]
        Cs = pd.DataFrame(nCs,columns=['id','Ag','Ab'])
        Cs['Ab'] = Cs.apply(lambda x: f"{x['id']}_{x['Ab']}", axis=1)
        Cs['Ag'] = Cs.apply(lambda x: f"{x['id']}_{x['Ag']}", axis=1)
        ex = np.concatenate([[f"{i}_{g}",f"{i}_{b}"]  for i,g,b in nCs])
        idxs = [df.index[df['Ab']==i].item() for i in ex]
        df = df.drop(idxs)
        df = pd.concat([df,Cs])
        for i, x in Cs.iterrows():
            p1 = x['Ab']
            p2 = x['Ag']
            logger.info(f" Contacts for : {i}, {p1}, {p2}")
            compute_aux_contact(p1,
                         p2, 
                         conf.OUTDIR, 
                         conf.expand_radius, 
                         conf.neighbor_dist,
                         conf.contact_mode, 
                         conf.contact_thres1, 
                         conf.contact_thres2,
                         device, 
                         smooth=False)    

    if len(Rs) > 0 :
        for i, x in Rs.iterrows():
            p = x['Ab']
            x = np.genfromtxt(f"{conf.OUTDIR}/{p}.xyz")
            logger.info(f" extracting area of interest for : {i}, {p}")
            compute_aux_coors(p,
                         conf.OUTDIR, 
                         conf.expand_radius, 
                         conf.neighbor_dist,
                         conf.contact_thres1, 
                         conf.contact_thres2,
                             device,x,
                         smooth=False)
    if len(Fs) > 0 :
        for i, x in Fs.iterrows():
            p = x['Ab']
            logger.info(f" extracting the full  surface area for : {i}, {p}")
            compute_aux_whole(p,
                         conf.OUTDIR,
                         conf.neighbor_dist,
            )        
    
else:
    print(f"set CONTACT =Ture or RESTRICT=True in {args.config} or provide regions column in {conf.CATALOG_PAIRS}" )
    exit()
        
    
if conf.DESC:
    p1s = df['Ab'].to_numpy()
    p2s = df['Ag'].to_numpy()
    logger.info(f" Generating embeddings for P1s: {p1s}")
    logger.info(f" Generating embeddings for P2s: {p2s}")
    fname = f"nn_desc.{conf.case}.npz"
    pdbs, x_desc, idxs_desc = gen_descriptors_contact(
        p1s, p2s, model, device, conf.params.get("rho_max"), conf.OUTDIR, fname)
    patch_sizes, pdbs_unique, starts, ends = get_desc_aux(pdbs)

# ---------------- All vs all search
# Load descriptors
# For each Ag epitope, compare against all other
if not conf.DESC and (conf.ALIGN or conf.SEARCH):
    fname = f"nn_desc.{conf.case}.npz"
    data = np.load(fname)
    pdbs = data.pdbs
    x_desc = data.x
    idxs_desc = data["idxs"]
    patch_sizes, pdbs_unique, starts, ends = get_desc_aux(pdbs)

if conf.SEARCH:
    # libraries
    library = df.apply(lambda x: f"{x['Ag']}.{x['Ab']}", axis=1).to_list()
    library_within = [np.load(os.path.join(conf.OUTDIR, x)) for x in df.apply(
        lambda x: f"{x['Ag']}_within.{x['Ab']}.npy", axis=1).to_list()]

    DIM = conf.params.get("num_filters")
    n = len(pdbs_unique)

    if conf.TARGET: targets = [conf.TARGET]
    else: targets = library
    
    
    summary = []
    for target in targets:

        target_,p2 = target.split(".")
        idxs_contact = np.load(os.path.join(conf.OUTDIR, f"{target_}_contacts.{p2}.npy"))
        st_target = time()
        o1, p1 = get_whole_molecule_desc(
            target_, model, conf.params.get("rho_max"), bs=128, device=device, outdir=conf.OUTDIR)
        o1 = torch.tensor(o1).to(device).view(len(o1), 1, DIM)
        if len(o1) > 20000:
            logger.info(" skip", target)
            continue
        

        mini_bs = int(64 * 1000 / len(o1))
        rho = p1.rho.cpu().numpy()
        list_indices = p1.list_indices.cpu().numpy()

        # Get inclusion matrix
        within = get_within(rho, 
                        list_indices, 
                        conf.neighbor_dist)
    
        logger.info(f" Searching for target {target_}")
        summary.extend(search(o1, rho, list_indices, within, library, library_within, pdbs_unique, patch_sizes,
                          pdbs, x_desc, idxs_desc, mini_bs, device, conf.nmin_pts, conf.expand_radius, conf.nmin_pts_library,
                              conf.thres, target, idxs_contact))
        # remove o1
        o1 = o1.cpu()
        del o1
        logger.info(f" Search time / dt = {time() - st_target:.3f}")

    df_hits = pd.DataFrame(summary, columns=conf.HIT_COLUMNS)
    df_hits["frac_expanded"] = df_hits["target_nhits"] / df_hits["target_nexpanded"]
    df_hits["frac_geo"] = np.sqrt(df_hits["frac_library_hits"] * df_hits["frac_expanded"])
    df_hits["target_expanded_indices"] = df_hits["target_expanded_indices"].apply(lambda x: list(x))
    df_hits["library_epitope_indices"] = df_hits["library_epitope_indices"].apply(
        lambda x: list(x))
    df_hits.to_csv(os.path.join(
        conf.OUTDIR_RESULTS, f"{conf.case}_top_hits.tsv"), sep="\t", index=False)

if conf.ALIGN:
    df_hits = pd.read_csv(os.path.join(
        conf.OUTDIR_RESULTS, f"{conf.case}_top_hits.tsv"), sep="\t")
    df_hits["target_expanded_indices"] = df_hits["target_expanded_indices"].apply(
        lambda x: np.asarray(eval(x), dtype=np.int32))
    df_hits["library_epitope_indices"] = df_hits["library_epitope_indices"].apply(
        lambda x: np.asarray(eval(x), dtype=np.int32))
    df_hits["id"] = -1
    conf.npatience = 100
    DIM = conf.params.get("num_filters")
    n = len(pdbs_unique)
    targets = df_hits["target"].drop_duplicates().tolist()
    librarys = df.apply(lambda x: f'{x["Ag"]}.{x["Ab"]}', axis=1).tolist()
    for target in targets:
        id_counter = 0
        st_target = time()
        logger.info(f" Aligning hits to target: {target}")
    
        target_, p2 = target.split(".")
        idxs_contact = np.load(os.path.join(conf.OUTDIR, f"{target_}_contacts.{p2}.npy"))
                                            
        if conf.SAVEPLY:
            ply_target = deepcopy(PlyData.read(
                os.path.join(conf.OUTDIR, f"{target_}.ply")))

        o_target, p_target = get_whole_molecule_desc(target_, model,
                                                 conf.params.get("rho_max"),
                                                 bs=128, device=device,
                                                 outdir=conf.OUTDIR)
        mini_bs = int(64 * 1000 / len(o_target))
        for library in librarys:
            df_ = df_hits[(df_hits["target"] == target) & (df_hits["library epitope"] == library)]
            nsample = min(10, len(df_))
            df_ = df_.sort_values("frac_geo", ascending=False).head(nsample)
            logger.info(f" library hit: {library}")
            for ie, entry in df_.iterrows():
                st = time()
                library_p1, library_p2 = entry['library epitope'].split(".")
                fname = os.path.join(conf.OUTDIR, f"{library_p1}_surface.npz")
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

                fname_target = os.path.join(conf.OUTDIR, f"{target_}_surface.npz")
                idxs_target = entry["target_expanded_indices"]

                if (len(idxs_target) < conf.nmin_pts) or (len(idxs_target) > 5000):
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
                    mask = (d < conf.thres).float()
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
                    if (nbad == conf.npatience):
                        logger.info(f" 1 / {i} loss_min {losses.min().item()} {idx1}")
                        break
                    if i == 0:
                        logger.info(f" 1 / {i} loss_min {losses.min().item()} {idx1}")
                xyz_patch, n = transform_library(x, y, z, T_zero, xyz_mean, aligner1, idx1, n)
                # collect stats
                df_hits.loc[ie, f"# hits repeat"] = (mask.sum(1) > 0).sum().item()
                df_hits.loc[ie, f"# pairs"] = npairs
                mask = mask.cpu().numpy()
                df_hits.loc[ie, f"aln1_score1"] = get_score1(xyz0, xyz_patch, mask, npairs)  
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

                logger.info(f" Time for aligning {library} to {target}: {time()-st:.2f}")
                if conf.SAVEPLY:
                    fname_library = os.path.join(conf.OUTDIR, f"{library_p1}.ply")
                    if not os.path.exists(fname_library):
                        continue

                    # -- create ply after aln1
                    # transform contact patch
                    fname = os.path.join(conf.OUTDIR, f"{library_p1}_surface.npz")
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
                    conf.OUTDIR_RESULTS, f"{id_counter}_{entry['library epitope']}_{target}.ply")
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
                    fname = os.path.join(conf.OUTDIR, f"{library_p1}.pdb")
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
                        conf.OUTDIR_RESULTS, f"{id_counter}_{entry['library epitope']}_{target}.pdb")
                    io = PDBIO()
                    io.set_structure(structure)
                    io.save(fname)
        logger.info(f" Total time for aligning hits to {target}: {time()-st_target:.2f}")
        if conf.SAVEPLY:
            out_target_fname = os.path.join(
                conf.OUTDIR_RESULTS, f"{conf.TARGET}_ref.ply")
            ply_target.write(out_target_fname)
        df_hits.to_csv(os.path.join(
                conf.OUTDIR_RESULTS, f"{conf.case}_top_hits_aligned.tsv"), sep="\t", index=False)

        # ---- Ranking vs. scores
        #df_ = df_hits[(~df_hits["aln1_score2"].isna())]
        #for rank in ["mean_desc_dist", "frac_expanded", "frac_library_hits", "frac_geo"]:
        #    plt.close()
        #    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        #    ax.scatter(df_[rank], df_["aln1_score2"], label="aln1")
        #    ax.set_ylim(
        #        [0, max(np.percentile(df_["aln1_score2"], 90) * 1.1, 1.0)])
        #    ax.set_xlabel(rank)
        #    ax.set_ylabel(
        #        f"mean feat dist of closest vertex pairs ({RLIM}) cutoff")
        #    ax.legend(loc="upper right")
        #    plt.savefig(os.path.join(
        #        conf.OUTDIR_RESULTS, f"{conf.case}_{rank}_vs_score2.png"), dpi=200, bbox_inches="tight")
        #    plt.close()
        #    ascending = True if rank == "mean_desc_dist" else False
        #    df__ = df_.sort_values(
        #        rank, ascending=ascending).reset_index(drop=True)[:50]
        #    iselect = df__["aln1_score2"] < 0.9
        #    nhit = np.sum(iselect)
        #    print(rank, nhit)

if conf.HEATMAP:
    df_hits = pd.read_csv(os.path.join(
        conf.OUTDIR_RESULTS, f"{conf.case}_top_hits_aligned.tsv"), sep="\t")
    cols = ["library epitope", "aln1_score2",
            "aln1_score3", "aln1_ncontact", "aln1_normal_score"]
    cols_new = ["library_epitope", "spatial_sim",
                "spatial_sim_desc", "ncontact", "normal_score"]
    df_hits.rename(
        columns={c: cn for c, cn in zip(cols, cols_new)}, inplace=True)
    fname = os.path.join(conf.OUTDIR_RESULTS, f"{conf.case}_top_hits_aligned.tsv")
    df_hits.to_csv(fname, sep="\t", index=False)


    iselect = df_hits.apply(lambda x: x["target"] not in x["library_epitope"], axis=1) & (df_hits["spatial_sim"] < 0.6) & (df_hits["spatial_sim"] >= 0.5)
    print(df_hits[iselect].sort_values(["target", "library_epitope", "frac_library_hits"], ascending=False)[["id"] + cols_new].drop_duplicates().head(50))
    rows = df["Ag"].to_numpy()
    cols = df.apply(lambda x:f'{x["Ag"]}.{x["Ab"]}', axis=1).to_numpy() # sorted(df_hits["target epitope"].drop_duplicates().tolist())
    if conf.RESTRICT:
        rows = cols            

    tally_dict = {}
    ii_cluster = None
    for rank in ["spatial_sim", "ncontact", "frac_library_hits", "frac_geo", "spatial_sim_desc", "normal_score"]:
        cmap = "rocket"
        fill_value = 0.
        vmin = 0.
        func = np.max
        # vmin = 0 if rank != "normal_score" else -1
        if rank == "ncontact":
            vmax, center = 100, 50
        elif rank in ["frac_library_hits", "frac_geo"]:
            vmax, center = 1.0, 0.5
        elif "spatial_sim_desc" == rank:
            vmax = 100.
            cmap = "rocket_r"
            fill_value = 1000
            func = np.min
        elif "spatial_sim" == rank:
            vmax = 1.0
            cmap = "rocket_r"
            fill_value = 100
            func = np.min
        elif "normal_score" == rank:
            vmax = 1.0
        tally = np.full((len(rows), len(cols)), fill_value, dtype=np.float32)
        for i, r in enumerate(rows):
            for j, c in enumerate(cols):
                df_ = df_hits[(df_hits["target"] == r) & (df_hits["library_epitope"] == c)]
                if len(df_):
                    q = func(df_[rank])
                    tally[i, j] = q if ~np.isnan(q) else fill_value
            if ii_cluster is None:
                Y = sch.linkage(tally, method='centroid')
                Z = sch.dendrogram(Y, orientation='left')
                ii_cluster = np.asarray(Z['leaves'])
            tally = tally[ii_cluster[:, None], ii_cluster]
            tally_dict[rank] = tally
            plt.close()
            figsize = (len(cols) * 0.5 + 2, len(rows) * 0.5 + 3)
            grid_kws = {"height_ratios": (.05, 0.90), "hspace": .3}
            f, (cbar_ax, ax) = plt.subplots(2, figsize=figsize, gridspec_kw=grid_kws)
            df_ = pd.DataFrame(tally, columns=cols[ii_cluster], index=rows[ii_cluster])                    
            ax = sns.heatmap(df_, ax=ax,
                             cbar_ax=cbar_ax,
                             cbar_kws={"orientation": "horizontal"},
                             vmin=vmin, vmax=vmax, cmap=cmap)#, center=center)
            fname = os.path.join(conf.OUTDIR_RESULTS, f"heatmap_{case}_{rank}.png")
            plt.savefig(fname, dpi=200, bbox_inches="tight")
            plt.close()
            mask = tally_dict["ncontact"] < nmin_pts
            tally = tally_dict["spatial_sim"]
            rank = "spatial_sim"
            cmap = "rocket_r"
            tally[mask] = 100.
            vmax = 1.0
            plt.close()
            figsize = (len(cols) * 0.5 + 2, len(rows) * 0.5 + 3)
            grid_kws = {"height_ratios": (.05, 0.90), "hspace": .3}
            f, (cbar_ax, ax) = plt.subplots(2, figsize=figsize, gridspec_kw=grid_kws)
            df_ = pd.DataFrame(tally, columns=cols[ii_cluster], index=rows[ii_cluster])
            ax = sns.heatmap(df_, ax=ax,
                             cbar_ax=cbar_ax,
                             cbar_kws={"orientation": "horizontal"},
                             vmin=0, vmax=vmax, cmap=cmap)#, center=center)
            fname = os.path.join(OUTDIR_RESULTS, f"heatmap_{case}_{rank}_masked.png")
            plt.savefig(fname, dpi=200, bbox_inches="tight")
            plt.close()

 

    
