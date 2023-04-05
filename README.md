# Surface Similarity Search via Geometric Deep Learning: SurfaceID

![alt text](https://github.com/Sanofi-GitHub/LMR-SurfaceID/blob/doc_sphinx/data/toc.png)

A protein can be represented in several forms, including its 1D sequence, 3D atom coordinates, and molecular surface. A protein surface contains rich structural and chemical features directly related to the protein’s function such as its ability to interact with other molecules. While many methods have been developed for comparing similarity of proteins using the sequence and structural representations, computational methods based on molecular surface representation are limited. Here, we describe “Surface ID”, a geometric deep learning system for high-throughput surface comparison based on geometric and chemical features.  Surface ID offers a novel grouping and alignment algorithm useful for clustering proteins by function, visualization, and in-silico screening of potential binding partners to a target molecule. Our method demonstrates top performance in surface similarity assessment, indicating great potential for protein functional annotation, a major need in protein engineering and therapeutic design.


## News

See [CHANGELOG.md](CHANGELOG.md) for detailed information about latest features.

## Setup
### Installation 

```bash
cd surfaceID
make install
```

### Test package

```bash
python3.9 -m venv sandbox
source sandbox/bin/activate
#untar the test_npz file and run:
python main.py --params data/config.yml --library data/test.tsv
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

#### Upload a file

Large Files are stored using git lfs

```bash
brew install git-lfs # or apt, yum, pacamn, aur etc.
git lfs track file
git add file
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

A copy of masif plugin python script is available under src/viz/. 
upon installing MaSIF plugin, append the path to this script in the ~/.pymolpluginsrc.py  

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
