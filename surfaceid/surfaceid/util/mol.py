import numpy as np
import os
from surfaceid.model.model import Model, clean_tensor, Mol, DNEG, DPOS, OUTDIR
import torch

class Mol_slim:
    """generates a molecule object what containts attributes for
       surface properties such as mesh edges, surface features,
       nehgbor list, and radial distance wrt patch center 
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
    """ generats a Mol object that contains attributes such as, 
        vertex ids and radial and angular distributions wrt patch center
        at the contact interface
    :param Mol: protein id
    :type Mol: str
    """

    def __init__(self, p, contacts=True, outdir=OUTDIR):
        p1, p2 = p
        fname = os.path.join(outdir, p1) + "_surface.npz"
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