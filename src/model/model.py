# parallelization in data loading does not seem to speed 
from time import time
import os
import subprocess
import numpy as np
import pandas as pd
import sys
from functools import partial
import tempfile
from multiprocessing import Pool, cpu_count
import shutil
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from adabelief_pytorch import AdaBelief
from scipy.linalg import block_diag
OUTDIR = "../data/20201107_SAbDab_masif"
DPOS = 1.5
DNEG = 5.0
torch.set_num_threads(cpu_count())


class Model(nn.Module):
    def __init__(self, rho_max=12.0, nbins_rho=5, nbins_theta=16, num_in=5, neg_margin = 10,
                share_soft_grid_across_channel=True, conv_type="sep",
                num_filters=80, add_center_pixel=False, min_sig=5e-2, nout=None,
                dropout=0.1, weight_decay=1e-2, lr=1e-3, nrotation=None, train=False):
        super(Model, self).__init__()
        self.num_in = num_in
        self.rho_max = rho_max
        self.nbins_rho=nbins_rho
        self.nbins_theta=nbins_theta
        self.neg_margin=neg_margin
        self.share_soft_grid_across_channel=share_soft_grid_across_channel
        self.conv_type=conv_type
        self.num_filters=num_filters
        self.add_center_pixel=add_center_pixel
        self.p = dropout
        self.weight_decay = weight_decay
        self.nrotation = nrotation if nrotation is not None else nbins_theta
        if nout is None:
            self.nout = num_filters

        # get rho
        drho = rho_max / nbins_rho
        rhos = np.arange(drho, rho_max + drho, drho, dtype=np.float32)

        if type(nbins_theta) is list:
            # precalc rotations
            dtheta = 2 * np.pi / self.nrotation
            thetas = np.arange(0, 2 * np.pi, dtheta, dtype=np.float32)
            self.dthetas = torch.tensor(thetas).reshape(1, len(thetas), 1, 1, 1)

            assert len(nbins_theta) == nbins_rho
            ts = []
            rs = []
            for i, (rho, nt) in enumerate(zip(rhos, nbins_theta)):
                dt = 2 * np.pi / nt
                t = np.arange(0, 2 * np.pi, dt, dtype=np.float32)
                ts.append(t)
                rs.append(np.full(len(t), rho, dtype=np.float32))
            thetas = torch.tensor(np.concatenate(ts))
            rhos = torch.tensor(np.concatenate(rs))
            self.ngrid = sum(nbins_theta)
        else:
            # precalc rotations
            dtheta = 2 * np.pi / nbins_theta
            thetas = np.arange(0, 2 * np.pi, dtheta, dtype=np.float32)
            self.dthetas = torch.tensor(thetas).reshape(1, len(thetas), 1, 1, 1)

            # get mesh
            thetas = torch.tensor(thetas).repeat(len(rhos))
            rhos = torch.repeat_interleave(torch.tensor(rhos), nbins_theta)
            self.ngrid = self.nbins_theta * self.nbins_rho
        assert self.ngrid == len(thetas)

        # repeat if sep
        if share_soft_grid_across_channel:
            # [B, 1, 1, Ng, Np]
            thetas = thetas.reshape(1, 1, 1, self.ngrid, 1)
            rhos = rhos.reshape(1, 1, 1, self.ngrid, 1)
            if add_center_pixel:
                a = np.full((1, 1, 1, 1, 1), drho / 2., dtype=np.float32)
                self.sig_rho_center = nn.parameter.Parameter(torch.tensor(a))
        else:
            # [B, 1, Nf, Ng, Np]
            thetas = torch.cat([thetas.reshape(1, 1, 1, self.ngrid, 1)] * num_filters, dim=1)
            rhos = torch.cat([rhos.reshape(1, 1, 1, self.ngrid, 1)] * num_filters, dim=1)
            thetas = thetas.reshape(1, 1, num_filters, self.ngrid, 1)
            rhos = rhos.reshape(1, 1, num_filters, self.ngrid, 1)
            if add_center_pixel:
                a = np.full((1, 1, num_filters, 1, 1), drho / 2., dtype=np.float32)
                self.sig_rho_center = nn.parameter.Parameter(torch.tensor(a))

        # Create vars
        self.thetas = nn.parameter.Parameter(thetas)
        self.rhos = nn.parameter.Parameter(rhos) 
        sigs_theta = torch.tensor(np.ones(rhos.size(), dtype=np.float32))
        sigs_rho = torch.tensor(np.full(rhos.size(), drho / 2., dtype=np.float32))
        self.sigs_theta = nn.parameter.Parameter(sigs_theta)
        self.sigs_rho = nn.parameter.Parameter(sigs_rho)
        self.min_sig = min_sig

        # conv            
        if conv_type == "sep":
            b = torch.zeros(1, num_filters * self.num_in, dtype=torch.float32)            
            Ws = [torch.empty(num_filters, self.ngrid, dtype=torch.float32) for _ in range(self.num_in)]
            for W in Ws:
                nn.init.xavier_normal_(W)
            W = torch.cat([W.unsqueeze(-1) for W in Ws], dim=-1).reshape(1, 1, num_filters, self.ngrid, self.num_in)
            if add_center_pixel:
                Ws_center = [torch.empty(num_filters, 1, dtype=torch.float32) for _ in range(self.num_in)]
                for W_center in Ws_center:
                    nn.init.xavier_normal_(W_center)
                W_center = torch.cat([W_center.unsqueeze(-1) for W_center in Ws_center], dim=-1).reshape(1, 1, num_filters, 1, self.num_in)
                self.W_center = nn.parameter.Parameter(W_center)
        else:
            b = torch.zeros(1, num_filters, dtype=torch.float32)            
            W = torch.empty(num_filters, self.ngrid * self.num_in, dtype=torch.float32)
            nn.init.xavier_normal_(W)
            W = W.reshape(1, 1, num_filters, self.ngrid, self.num_in)            
            if add_center_pixel:
                W_center = torch.empty(num_filters, self.num_in, dtype=torch.float32)
                nn.init.xavier_normal_(W_center)
                W_center = W_center.reshape(1, 1, num_filters, 1, self.num_in)            
                self.W_center = nn.parameter.Parameter(W_center)
        self.W = nn.parameter.Parameter(W)            
        self.b = nn.parameter.Parameter(b)

        # final MLP
        self.gelu = nn.GELU()
        self.dropout1 = nn.Dropout(self.p)
        self.dropout2 = nn.Dropout(self.p)
        if conv_type == "sep":
            self.fc1 = nn.Linear(num_filters * self.num_in, num_filters)
        else:
            self.fc1 = nn.Linear(num_filters, num_filters)
        self.fc2 = nn.Linear(num_filters, num_filters)

        # opt
        self.train_model = train
        if self.train_model:
            decay = ["weight", "W"]
            optimizer_grouped_parameters = [
                {'params': [p for n, p in self.named_parameters() if any(nd in n for nd in decay)], 'weight_decay': self.weight_decay},
                {'params': [p for n, p in self.named_parameters() if not any(nd in n for nd in decay)], 'weight_decay': 0.0}
            ]
            # self.optimizer = optim.AdamW(optimizer_grouped_parameters, lr=lr)
            self.optimizer = AdaBelief(optimizer_grouped_parameters, lr=lr)

        return

    def forward(self, x, rho, theta, mask, pos=None, neg=None, calc_loss=True):
        B = x.size()[0]
        npoints = x.size()[1]
        mask[rho > self.rho_max] = 0.
        rho = rho.view(B, 1, 1, 1, npoints) # .contiguous()
        theta = theta.view(B, 1, 1, 1, npoints) # .contiguous()
        mask = mask.view(B, 1, 1, 1, npoints) # .contiguous()
        x = x.view(B, 1, 1, 1, npoints, self.num_in) # .contiguous()        

        if self.add_center_pixel:
            # calculate weight [B, 1, 1, 1, Np]
            arg_center = torch.square(rho / torch.clamp(self.sig_rho_center, min=self.min_sig))
            weight_center = (torch.exp(-arg_center) + 1e-12) * mask
            norm_center = weight_center.sum(dim=-1, keepdim=True) # over all pos
            weight_center = weight_center / (norm_center + 1e-12)
            # map to grid
            z_center = (weight_center.unsqueeze(-1) * x).sum(dim=-2) # sum over points; out [B, Nrot=1, Nf=1, Ng=1, num_in]
            h_center = (z_center * self.W_center).sum(dim=-2) # [B, 1, 1, num_in]

        arg0 = torch.square((rho-self.rhos) / torch.clamp(self.sigs_rho, min=self.min_sig))
        theta_ = torch.remainder(theta + self.dthetas, 2 * np.pi)
        arg1 = torch.square((theta_ - self.thetas) / torch.clamp(self.sigs_theta, min=self.min_sig))
        arg = arg0 + arg1
        weight = (torch.exp(-arg) + 1e-12) * mask
        norm = weight.sum(dim=-1, keepdim=True) # over all pos
        weight = weight / (norm + 1e-12)
        if torch.isnan(weight).any():
            print(weight)
            assert False

        # map features
        z = (weight.unsqueeze(-1) * x).sum(dim=-2) # sum over points; out [B, Nrot, NF, Ng, num_in]

        # perform conv
        h = (z * self.W).sum(dim=-2) # [B, Nrot, NF, num_in]
        if self.add_center_pixel:
            h += h_center

        if self.conv_type == "sep":
            o = h.view(B, self.nrotation, self.num_filters * self.num_in) # .contiguous() # [B, NF * num_in] 
        else:
            o = h.sum(dim=-1) # [B, Nrot, NF]

        # rotation pooling
        o, _ = torch.max(o, dim=1)
        o = o + self.b

        # non-linear
        o = self.gelu(o)
        o = self.gelu(self.fc1(self.dropout1(o)))
        o = self.fc2(self.dropout2(o))

        if calc_loss:
            d_neg, d_pos, loss = self.get_dist(o, pos, neg)
        else:
            loss = d_neg = d_pos = None

        return o, loss, d_pos, d_neg

    def get_dist(self, o, pos_mask, neg_mask):
        B = len(o)
        o1 = o.view(B, 1, self.nout) # .contiguous() # [B, 1, nout]
        o2 = o.view(1, B, self.nout) # .contiguous()
        d = torch.norm(o1 - o2, dim=2)
        d_pos = d[pos_mask == 1.]
        d_neg = d[neg_mask == 1.]
        d_pos_loss = torch.clamp(d_pos, min=0.)
        d_neg_loss = torch.clamp(self.neg_margin - d_neg, min=0.)
        pos_mean, pos_std = torch.mean(d_pos_loss), torch.std(d_pos_loss)
        neg_mean, neg_std = torch.mean(d_neg_loss), torch.std(d_neg_loss)
        loss = pos_mean + pos_std + neg_mean + neg_std

        return d_neg, d_pos, loss        

    def get_dist_light(self, o, pos_mask, neg_mask, bs=128):
        B = len(o)
        d = np.zeros((B, B), dtype=np.float32)
        o2 = o.view(1, B, self.nout) # .contiguous()
        for st in range(0, B, bs):
            e = min(st+bs, B)
            o1 = o[st:e].view(e-st, 1, self.nout) # .contiguous() # [B, 1, nout]
            d_ = torch.norm(o1 - o2, dim=2)
            d[st:e] = d_.detach().cpu().numpy()
        d_pos = d[pos_mask == 1.]
        d_neg = d[neg_mask == 1.]
        d_pos_loss = np.maximum(d_pos, 0.)
        d_neg_loss = np.maximum(self.neg_margin - d_neg, 0.)
        pos_mean, pos_std = np.mean(d_pos_loss), np.std(d_pos_loss)
        neg_mean, neg_std = np.mean(d_neg_loss), np.std(d_neg_loss)
        loss = pos_mean + pos_std + neg_mean + neg_std

        return d_neg, d_pos, loss


def clean_tensor(device, x, rho, theta, mask):
    x[np.isnan(x)] = 0
    rho[np.isnan(rho)] = 0
    theta[np.isnan(theta)] = 0
    x = np.maximum(x, -1.0)
    x = np.minimum(x, 1.0)
    x = torch.tensor(x, dtype=torch.float32).to(device)
    rho = torch.tensor(rho, dtype=torch.float32).to(device)
    theta = torch.tensor(theta, dtype=torch.float32).to(device)
    mask = torch.tensor(mask, dtype=torch.float32).to(device)

    return x, rho, theta, mask


def sample_patches(p, nsample_per_mol, data_type, rmax):
    idx = np.random.randint(0, p.npoints, 1)[0]

    # patch
    r = p.rho[idx]
    idxs_patch = p.list_indices[idx]
    idxs_patch = idxs_patch[(idxs_patch != -1) & (r < rmax)]
    idxs_patch = np.unique(np.random.choice(idxs_patch, nsample_per_mol, replace=True))

    # distance
    n = len(idxs_patch)
    d = np.full((n, n), 100, dtype=np.float32)
    for i, idx1 in enumerate(idxs_patch):
        for j, idx2 in enumerate(idxs_patch):
            idxs1 = p.list_indices[idx1]
            z = np.where(idxs1==idx2)[0]
            if len(z) == 1:
                d[i, j] = p.rho[idx1][z[0]]
    pos = np.triu(np.ones_like(d), 1) * (d < DPOS)
    neg = np.triu(np.ones_like(d), 1) * (d > DNEG)
    x = np.concatenate([p.x[p.list_indices[idx]].reshape(1, 200, 5) for idx in idxs_patch])

    return (x, p.rho[idxs_patch], p.theta[idxs_patch], p.mask[idxs_patch],
            pos, neg)


class Mol:
    def __init__(self, fname):
        try:
            x = np.load(fname)
            self.fname = fname
            self.list_indices = x["list_indices"]    
            self.rho = x["rho"]
            self.theta = x["theta"]
            self.mask = x["mask"]
            self.x = x["x_initial"]
            self.npoints = len(x)            
            self.status = "good"
        except:
            self.status = "bad"


def get_batch(device, nsample_per_mol, data_type, num_mol, mols, rmax, rho_max):
    ps = np.random.choice(mols, num_mol, replace=False)
    patches = [sample_patches(p, nsample_per_mol, data_type, rmax) for p in ps]
    # parallel veresion not as good
    # func = partial(sample_patches, nsample_per_mol=nsample_per_mol,
    #                data_type=data_type, rmax=rmax)
    # patches = pool.map(func, ps)
    x, rho, theta, mask, poss, negs = zip(*patches)
    x_ = np.concatenate(x)
    rho_ = np.concatenate(rho)
    theta_ = np.concatenate(theta)
    mask_ = np.concatenate(mask)

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

    n = len(x_)
    pos = block_diag(*poss)
    neg_intra = block_diag(*negs)
    blocks = block_diag(*[np.ones_like(x) for x in negs])
    neg_inter = np.ones((n, n), dtype=np.float32) - blocks
    neg = np.triu(neg_intra + neg_inter, 1)

    return (x_, rho_, theta_, mask_,
            torch.tensor(pos).to(device), torch.tensor(neg).to(device))


def eval_dataset(model, dataset, device):
    """streamlined"""
    model.eval()
    with torch.set_grad_enabled(False):
        # d_poss = []
        # d_negs = []
        # loss_total = 0
        poss = []
        negs = []
        os = []
        for x, rho, theta, mask, pos, neg in dataset:
            x = x.to(device)
            rho = rho.to(device)
            theta = theta.to(device)
            mask = mask.to(device)
            pos = pos.to(device)
            neg = neg.to(device)
            o, loss, d_pos, d_neg = model(x, rho, theta, mask, pos, neg, calc_loss=True)
            # d_poss.append(d_pos.cpu().numpy())
            # d_negs.append(d_neg.cpu().numpy())
            poss.append(pos.cpu().numpy())
            negs.append(neg.cpu().numpy())
            os.append(o)
        #     loss_total += loss.item()
        # d_pos = np.concatenate(d_poss)
        # d_neg = np.concatenate(d_negs)
        o = torch.cat(os)
        pos_mask = np.triu(block_diag(*poss), 1)
        ones = 1 - block_diag(*[np.full((len(o), len(o)), 1) for o in os])
        neg_mask = np.triu(block_diag(*negs) + ones, 1)
        d_neg, d_pos, loss = model.get_dist_light(o, pos_mask, neg_mask)

    return loss, d_pos, d_neg


def gen_test_val(ps, device='cpu'):
    # sample one patch from one molecule
    fs = [os.path.join(OUTDIR, p) + "_surface.npz" for p in ps]
    mols = [Mol(f) for f in fs if os.path.exists(f)]
    mols = [m for m in mols if m.status == "good"]
    packets = [get_batch(device, 200, "all", 1, [m], 12.0, 12.0) for m in mols]
    # packet -- x_, rho_, theta_, mask_, pos_, neg_

    return packets


if __name__ == "__main__":
    pdbs_train = pd.read_csv("20201107_SAbDab_masif_train_pdbs.tsv", sep="\t")["pdb"].to_numpy()
    pdbs_test = pd.read_csv("20201107_SAbDab_masif_test_pdbs.tsv", sep="\t")["pdb"].to_numpy()
    pdbs_val = pd.read_csv("20201107_SAbDab_masif_val_pdbs.tsv", sep="\t")["pdb"].to_numpy()

    # preload test-val data, unrestricted to epitope
    val_data = gen_test_val(pdbs_val)
    test_data = gen_test_val(pdbs_test)    

    savedir = "final"
    os.makedirs(savedir, exist_ok=True)
    max_ckpts = 1000
    patience = 500
    expts = [
    ["final_000",
        {"rho_max": 12., "nbins_rho": 5, "nbins_theta": 16, "num_in": 5,
             "neg_margin": 10, "add_center_pixel": True,
             "share_soft_grid_across_channel": True,
             "conv_type": "sep", "num_filters": 1024, "weight_decay": 1e-2,
             "dropout": 0.1, "min_sig": 5e-2, "lr": 5e-4},
        {"grad_norm_clip": 5., "nbatch_ckpt": 500, "max_ckpts": max_ckpts,
            "patience": patience, "data": "all", "num_mol": 2, "nsample_per_mol": 64,
            "nmol_per_ckpt": 100, "rmax": 8.0}],
    ["final_001",
        {"rho_max": 4.5, "nbins_rho": 5, "nbins_theta": 16, "num_in": 5,
             "neg_margin": 10, "add_center_pixel": True,
             "share_soft_grid_across_channel": True,
             "conv_type": "sep", "num_filters": 512, "weight_decay": 1e-2,
             "dropout": 0.1, "min_sig": 5e-2, "lr": 5e-4},
        {"grad_norm_clip": 5., "nbatch_ckpt": 500, "max_ckpts": max_ckpts,
            "patience": patience, "data": "all", "num_mol": 2, "nsample_per_mol": 150,
            "nmol_per_ckpt": 100, "rmax": 12.0}],
    ["final_002",
        {"rho_max": 6.0, "nbins_rho": 5, "nbins_theta": 16, "num_in": 5,
             "neg_margin": 10, "add_center_pixel": True,
             "share_soft_grid_across_channel": True,
             "conv_type": "sep", "num_filters": 512, "weight_decay": 1e-2,
             "dropout": 0.1, "min_sig": 5e-2, "lr": 5e-4},
        {"grad_norm_clip": 5., "nbatch_ckpt": 500, "max_ckpts": max_ckpts,
            "patience": patience, "data": "all", "num_mol": 1, "nsample_per_mol": 200,
            "nmol_per_ckpt": 100, "rmax": 12.0}]
    ]

    expts_train = []
    for prefix, params, train_params in expts:
        start = time()
        print(prefix)
        path = os.path.join(savedir, f"model_{prefix}.pth")
        model = Model(**params)
        device = 'cpu'
        if torch.cuda.is_available():
            device = torch.cuda.current_device()        
            model = model.to(device)
            model.dthetas = model.dthetas.to(device)
        if prefix in expts_train:
            params["train"] = True
            train_loss = []
            val_loss = []
            val_loss_best = np.infty
            nckpt_bad = 0
            # with Pool(cpu_count()-1) as pool:
            for ckpt in range(train_params["max_ckpts"]):
                # train
                st = time()
                model.train()
                train_loss_tmp = 0
                ps = np.random.choice(pdbs_train, train_params["nmol_per_ckpt"], replace=False)
                fs = [os.path.join(OUTDIR, p) + "_surface.npz" for p in ps]
                # # parallel
                # fs = [f for f in fs if os.path.exists(f)]
                # mols = pool.map(Mol, fs)
                mols = [Mol(f) for f in fs if os.path.exists(f)]
                mols = [m for m in mols if m.status == "good"]
                nneg = npos = 0
                for j in range(train_params["nbatch_ckpt"]):
                    model.optimizer.zero_grad()
                    x_, rho_, theta_, mask_, pos_, neg_ = \
                        get_batch(device, train_params["nsample_per_mol"], train_params["data"],
                                  train_params["num_mol"], mols, train_params["rmax"], params["rho_max"])
                    npos += pos_.sum().item()
                    nneg += neg_.sum().item()                    
                    o, loss, _, _ = model(x_, rho_, theta_, mask_, pos_, neg_, calc_loss=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_params["grad_norm_clip"])
                    model.optimizer.step()
                    train_loss_tmp += loss.item()
                train_loss_tmp /= train_params["nbatch_ckpt"]
                train_loss.append(train_loss_tmp)
                print(f"{ckpt} train: {train_loss_tmp:.3f}")
                print(f"{ckpt} avg pos: {npos / train_params['nbatch_ckpt'] / train_params['num_mol']}")
                print(f"{ckpt} avg neg: {nneg / train_params['nbatch_ckpt'] / train_params['num_mol']}")
                print(f"{ckpt} train dt: {time()-st:.3f}")                

                # validation
                st = time()
                model.eval()
                val_loss_tmp, d_pos, d_neg = eval_dataset(model, val_data, device)
                print(f"{ckpt} val: {val_loss_tmp:.3f}")
                print(f"{ckpt} {time()-st:.3f}")
                val_loss.append(val_loss_tmp)
                if (val_loss_tmp - 0.05) < val_loss_best:
                    if val_loss_tmp < val_loss_best:
                        val_loss_best = val_loss_tmp
                        nckpt_bad = 0
                        print("update best")
                    else:
                        nckpt_bad += 1
                    torch.save(model.state_dict(), path)                    
                else:
                    nckpt_bad += 1

                patience_left = train_params["patience"] - nckpt_bad
                print(f"patience left: {patience_left}")

                if nckpt_bad > train_params["patience"]:
                    break
                print()

                # plot curve
                plt.close()
                fig, ax = plt.subplots(1, 1, figsize=(7, 5))
                xs = range(len(train_loss))
                ax.grid(True)
                ax.plot(xs, train_loss, color="black", lw=1, label="train loss")
                ax.plot(xs, val_loss, color="red", lw=1, label="val loss")
                ax.set_xlabel("# ckpt")
                ax.set_ylabel("loss")
                ax.set_xlim([0, ckpt+1])
                ax.set_ylim([0, 10.0])
                ax.legend(loc="upper right")
                plt.savefig(os.path.join(savedir, f"loss_{prefix}.png"), dpi=200, bbox_inches="tight")
                plt.close()             
        dt = time() - start              
        print(f"{dt:.3f}")

        # test --- test dataset
        model.load_state_dict(torch.load(path))            
        model.eval()
        loss, d_pos, d_neg = eval_dataset(model, test_data, device)
        print(f"{loss:.3f}")

        # plot
        bins = np.arange(0, 15, 0.05)
        plt.close()
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        ax.set_title(prefix)
        ax.hist(d_pos, bins=bins, alpha=0.5, label=f"self-neighbor [0.5, 1.5]")
        ax.hist(d_neg, bins=bins, alpha=0.5, label=f"all-else")
        ax.legend(loc="upper right")
        ax.set_xlabel("Descriptor dist.")
        ax.set_xlim([0, 15])
        ax.set_ylabel("Count")
        plt.savefig(os.path.join(savedir, f"descr_dist_count_{prefix}.png"), dpi=200, bbox_inches="tight")
        plt.close()

        plt.close()
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        ax.set_title(prefix)
        ax.hist(d_pos, bins=bins, alpha=0.5, density=True, label=f"self-neighbor [0.5, 1.5]")
        ax.hist(d_neg, bins=bins, alpha=0.5, density=True, label=f"all-else")
        ax.legend(loc="upper right")
        ax.set_xlabel("Descriptor dist.")
        ax.set_xlim([0, 15])
        ax.set_ylabel("Density")
        plt.savefig(os.path.join(savedir, f"descr_dist_density_{prefix}.png"), dpi=200, bbox_inches="tight")
        plt.close()

        # prec recall
        ntotal_pos = len(d_pos)
        recall = 0
        c = 1
        recalls = [10, 25, 50, 60, 70, 80, 90, 95, 99]
        thress = np.percentile(d_pos, recalls)
        for thres, recall in zip(thress, recalls):
            fp = (d_neg < thres).sum()
            tp = (d_pos < thres).sum()
            prec = tp / (fp + tp)
            recall = tp / ntotal_pos
            r = f"{prec:.3f}/{recall:.3f}"
            print(r)
            print(f"{thres:.3f}")
        del model
        print()
