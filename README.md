# Surface Similarity Search via Geometric Deep Learning

## Setup
### Installation 
```
cd surface_id
pip install poetry 
cd surfaceID
make install
```

### Test package

```
python3.9 -m venv sandbox
source sandbox/bin/activate
```

## Preprocessing 
Files can be loaded with gitlfs and will populate data directory

### Upload a file
```bash
brew install git-lfs # or apt, yum, pacamn, aur etc.
git lfs *
git add *
git commit -m "wip: adding train data sample"
```

### Download a file
```bash
git lfs pull
GIT_TRACE=1 git lfs fetch # debug

## Output of the code:


### Train
Assume `data/20201107_SAbDab_masif` data are available. To train the model, simply run 
```
python model/models.py
```

## Visualization

A copy of masif plugin python script is available here.