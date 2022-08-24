# Surface Similarity Search via Geometric Deep Learning: SurfaceID

![alt text](https://github.com/Sanofi-GitHub/LMR-SurfaceID/blob/main/data/toc.png?raw=true)


## News

See [CHANGELOG.md](CHANGELOG.md) for detailed information about latest features.

## Setup
### Installation 
```bash
cd surface_id
pip install poetry 
cd surfaceID
make install
```

### Test package

```bash
python3.9 -m venv sandbox
source sandbox/bin/activate
python main.py
```

## Data
### Preprocessing 
Files can be loaded with gitlfs and will populate data directory.
MaSIF publication's original pipeline is used (with modification) 
for generating a surface representation per moleclue.

For each molecule, the modified versions of the MaSIF scripts are run to generate surface representation files. These files are "collated" into a file `{pdb}_{chain}_suface.npz` that contains the triangular mesh and properties at each vertex and the `{pdb}_{chain}.ply/pdb` files for surface and PDB files. Each `*_surface.npz` file contains the following entries: 

- pos [num_vertices, 3]; float
- edge_index [2, num_edges]; int
- x_{embedding_name} [num_vertices, d_embedding]; float;
- rho [num_vertices, max_num_vertices_per_patch]; float; padded zero
- theta [num_vertices, max_num_vertices_per_patch]; float; padded zero
- x_local [num_vertices, max_num_vertices_per_patch, d_embedding_local]; float; padded zero; hand-engineered 5 numbers
- mask [num_vertices, max_num_vertices_per_patch]; bool; padding mask
- list_indices [num_vertices, max_num_vertices_per_patch]; int; padded -1 for masked
- face [3, num_faces]; int
- normals [num_vertices, 3]; float
- iface [num_vertices, 1]; float; Value eq 1 if the vertex is in contact with another molecule in the same PDB file, where contact is defined by solvent excluded region with/without other molecules.

#### 20201107_SAbDab data set
A copy of the DB pdb files are obtained by Yu Qiu and stored as `data/20201107_SAbDab`. Some catalogs generated using `gen_SAbDab_catalog.py`.

Large Files are stored using git lfs

#### Upload a file
```bash
brew install git-lfs # or apt, yum, pacamn, aur etc.
git lfs *
git add *
git commit -m "wip: adding train data sample"
```

#### Download a file
```bash
git lfs pull
GIT_TRACE=1 git lfs fetch # debug
```


## Train
Assume `data/20201107_SAbDab_masif` data are available. To train the model, simply run 
```
python model/models.py
```

## Visualization

A copy of masif plugin python script is available here.

# Reference

Reference to cite when you use Faiss in a research paper:

```
@article{Rihai,
  title={Surface ID: A Geometry-aware System for Protein Molecular Surface Comparison},
  author={Saleh Riahi1&*, Jae Hyeon Lee2&#, Taylor Sorenson2, Shuai Wei1†, Sven Jager3, Reza Olfati-Saber2, Anna Park1, Maria Wendt1, Hervé Minoux4*, Yu Qiu1*},
  journal={Nature Communications},
  volume={7},
  number={3},
  pages={535--547},
  year={2022},
  publisher={Nature}
}
```