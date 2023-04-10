# Surface Similarity Search via Geometric Deep Learning: SurfaceID

![alt text](https://github.com/Sanofi-GitHub/LMR-SurfaceID/blob/main/docs/toc.png)

A protein can be represented in several forms, including its 1D sequence, 3D atom coordinates, and molecular surface. A protein surface contains rich structural and chemical features directly related to the protein’s function such as its ability to interact with other molecules. While many methods have been developed for comparing similarity of proteins using the sequence and structural representations, computational methods based on molecular surface representation are limited. Here, we describe “Surface ID”, a geometric deep learning system for high-throughput surface comparison based on geometric and chemical features.  Surface ID offers a novel grouping and alignment algorithm useful for clustering proteins by function, visualization, and in-silico screening of potential binding partners to a target molecule. Our method demonstrates top performance in surface similarity assessment, indicating great potential for protein functional annotation, a major need in protein engineering and therapeutic design.




## Setup
### requirements. 
Assuming that MaSIF and its packages are installed, or the surface npz files are provided, surface ID requirements can be installed with:

```bash
pip install -r requirements.txt
```



## Data
### Preprocessing 
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



## Visualization of the SurfaceID search outputs.

A copy of masif plugin python script is available under src/pymol_plugin . 
upon installing MaSIF plugin, append the path to this script in the ~/.pymolpluginsrc.py  
If the SurfaceID search is conducted and the "ALIGNED & SAVEPLY" are set to "TRUE" in the config.yml file, the hit surface area on target and candidate protesins can be illustrated by "sidloadply <candidate.target.ply>" for candidate hit and "sidloadply <target_ref.ply>" for the patches on target.

# Reference

Please cite the following work:

```
S Riahi, J Hyeon Lee, T Sorenson, S Wei1, S Jager, R Olfati-Saber, A Park, M Wendt, H Minoux, Y Qiu;
Surface ID: A Geometry-aware System for Protein Molecular Surface Comparison;
Bioinformatics, X,X,2023 

```
