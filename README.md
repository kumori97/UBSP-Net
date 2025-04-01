# **Underclothing Body Shape Perception Network for Parametric 3D Human Reconstruction** ( for review)


## Prerequisites
1. Cuda 10.0
2. Cudnn 7.6.5
3. Kaolin 0.1 (https://github.com/NVIDIAGameWorks/kaolin) - for SMPL registration
4. MPI mesh library (https://github.com/MPI-IS/mesh)
5. Trimesh
6. Python 3.7
9. SMPL pytorch from https://github.com/gulvarol/smplpytorch.

## Download relational models
1. Download vsmpl model:  https://pan.baidu.com/s/1uLR7AI0LbJMTKvsKBTfP7A?pwd=mh92 and put them in `assets` folder
2. Download smpl pkl model: https://smpl.is.tue.mpg.de/ and put them in `assets/smpl` folder
## Run  demo
1. Test UBSP-Net: `python test_single_UBSPNet.py `
2. SMPL register to prediction: `python smpl_model_register.py`
3. SMPLD register to prediction:`python smpld_model_register.py`

## Train UBSP-Net
1. Building  dataset:`python caculate_correspondences.py `
2. Train: `python train.py `

Due to the large size of the dataset, we only provided  processeddataset samples (assets\CAPE_Dataset_Sampled_10000) for review purposes. You can use `python caculate_correspondences.py` to build the full dataset from the CAPE dataset.


