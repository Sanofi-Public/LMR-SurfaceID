import pymesh
import os
import subprocess
import numpy as np
import pandas as pd
import sys
import tempfile
from multiprocessing import Pool, cpu_count
import shutil
PRECOMP_DIR = sys.argv[1]
OUTDIR = sys.argv[2] # S3 bucket where the results will be saved.

def get_edges_from_faces(face):
    edges1 = face[:2]
    edges2 = face[1:]
    edges3 = face[[0, 2], :]
    edges = np.concatenate([edges1, edges2, edges3], axis=1).T
    # TODO: redundant edges
    edges = np.unique(edges, axis=0).T

    return edges


def save_masif_surface(pdb, chain):
    # load the mesh to extract faces and vertices
    ply_fn = os.path.join(PRECOMP_DIR,'plys',f"{pdb}_{chain}.ply")
    mesh = pymesh.load_mesh(ply_fn) 
    face = mesh.faces.T
    pos = mesh.vertices
    n1 = mesh.get_attribute("vertex_nx").reshape(len(pos), 1)
    n2 = mesh.get_attribute("vertex_ny").reshape(len(pos), 1)
    n3 = mesh.get_attribute("vertex_nz").reshape(len(pos), 1)
    normals = np.concatenate([n1, n2, n3], axis=1)
    edge_index = get_edges_from_faces(face) 


    precomp_dir_pdb_chain = os.path.join(PRECOMP_DIR,f"{pdb}_{chain}")

    x_local = np.load(os.path.join(precomp_dir_pdb_chain, "p1_input_feat.npy"))
    x_initial = x_local[:, 0, :]
    rho = np.load(os.path.join(precomp_dir_pdb_chain, "p1_rho_wrt_center.npy"))
    theta = np.load(os.path.join(precomp_dir_pdb_chain, "p1_theta_wrt_center.npy"))
    mask = np.load(os.path.join(precomp_dir_pdb_chain, "p1_mask.npy")) 
    list_indices = np.load(os.path.join(precomp_dir_pdb_chain, "p1_list_indices.npy"), allow_pickle=True) 
    list_indices = [np.concatenate([x, np.ones(200 - len(x), dtype=int) * -1]).reshape(1, 200) if len(x) < 200 else np.asarray(x).reshape(1, 200) for x in list_indices]
    list_indices = np.concatenate(list_indices, axis=0)
    iface = np.load(os.path.join(precomp_dir_pdb_chain, "p1_iface_labels.npy"))
    keys = ["face", "pos", "normals", "edge_index", "x_local", "x_initial", "rho", "theta", "mask", "list_indices","iface"]
    vals = [face, pos, normals, edge_index, x_local, x_initial, rho, theta, mask, list_indices,iface]
    store = {k: v for k, v in zip(keys, vals)}
    fname = f"{pdb}_{chain}_surface.npz"
    done = False
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_fname = os.path.join(tmp_dir, fname)
        np.savez(tmp_fname, **store)
        cmd = f"mv {tmp_fname} {OUTDIR}/"
        subprocess.call(cmd, shell=True)

        # confirm the export
        fname_remote = os.path.join(OUTDIR, fname)
        cmd = f"cp {fname_remote} {tmp_dir}/"  
        subprocess.call(cmd, shell=True)
        x = np.load(tmp_fname)
        assert all([np.all(np.equal(x[k], v)) for k, v in zip(keys, vals)])
        done = True

    # remove current
    if done:
        print(fname, done)
        # shutil.rmtree(precomp_dir_pdb_chain)
        # os.remove(ply_fn)

    return

f = pd.read_csv(sys.argv[3],sep=',')
for i,pdb in enumerate(f['id']):
    print(pdb,f['chain'][i])
    save_masif_surface(pdb.split('.pdb')[0],f['chain'][i])
