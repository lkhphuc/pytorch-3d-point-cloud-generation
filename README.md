# Pytorch: Learning Efficient Point Cloud Generation for Dense 3D Object Reconstruction
A Pytorch implementation of the paper: Learning Efficient Point Cloud Generation
for Dense 3D Object Reconstruction 

- Article: https://medium.com/@lkhphuc/create-3d-model-from-a-single-2d-image-in-pytorch-917aca00bb07
- Original Website: https://chenhsuanlin.bitbucket.io/3D-point-cloud-generation
- Original TF implementation: https://github.com/chenhsuanlin/3D-point-cloud-generation

--------------------------------------

## Training/evaluating the network

### Prerequisites
This code is developed with Python3 (`python3`). Pytorch 0.4+ is required.

### Dataset
(Provided in TF's repo)
The dataset (8.8GB) can be downloaded by running the command
```
wget https://cmu.box.com/shared/static/s4lkm5ej7sh4px72vesr17b1gxam4hgy.gz
```
This file includes:
- Train/test split files (from [Perspective Transformer Nets](https://github.com/xcyan/nips16_PTN))
- Input RGB images (from [Perspective Transformer Nets](https://github.com/xcyan/nips16_PTN))
- Pre-rendered depth images for training
- Ground-truth point clouds of the test split (densified to 100K points)

After downloading, run `tar -zxf s4lkm5ej7sh4px72vesr17b1gxam4hgy.gz` under the main directory. The files will be extracted to the `data` directory.
(Please also cite the relevant papers if you plan to use this dataset package.)

### Running the code
The following scripts gives examples for running the code.
- Pretraining the network: `scripts/train-stg1.sh`
- Fine-tuning with joint 2D optimization: `scripts/train-stg2.sh`
- Evaluating on the test set: `scripts/evaluate.sh`
- Computing the error metrics: `scripts/evaluate_dist.sh`

Checkpoints are stored in `models/${EXPERIMENTATION}`, summaries are stored in `runs/_${EXPERIMENTATION}`, and evaluated point clouds are stored in `results_${GROUP}`.
The list of optional arguments can be found by executing `python3 train-stg1.py --help`.

--------------------------------------

## Rendering ground-truth depth images
(Provided in TF's repo)
We provide the code to render depth images for supervision.

### Prerequisites
This code requires the following:
- [Blender](https://www.blender.org/) as the rendering engine. This code was developed with Blender 2.78.
  After installation, please make sure the command `blender` is callable (use `which blender` to check installation).
- The [OpenEXR Python binding](http://www.excamera.com/sphinx/articles-openexr.html) for .exr to .mat file conversion.

### Dataset
The raw ShapeNet dataset can be downloaded [here](https://www.shapenet.org/).
This rendering code was developed to use ShapeNetCore v2. (The provided depth images were rendered from ShapeNetCore v1.)

### Running the code
Under `render`, run `./run.sh 03001627 8` to render depth images for fixed and arbitrary viewpoints, and convert them to .mat files. This will convert all objects in the ShapeNet chair category (03001627) with 8 fixed viewpoints.
The rendered files will be stored in the `output` directory.

--------------------------------------

## Creating densified point clouds of CAD models for evaluation
(Provided in TF's repo)
We also provide the code to densify the vertices of CAD models to a specified number. This code can be run independently; only the ShapeNet dataset is required.
It repeats the process of adding a vertex to the center of the longest edge of the triangular mesh and subsequently re-triangulating the mesh. This will create (generally) uniformly densified CAD models.
<p align="center"><img src="densify/example.png" width=400></p>

### Running the code
Under `densify`, run `./run.sh 03001627` to run densification. The densified CAD models will be stored in the `output` directory.
