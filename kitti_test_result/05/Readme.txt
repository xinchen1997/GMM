*_every_tf is the transformation for every two frames
*_seq_tf is the sequential transformation
MAX_ITER = 30
TOL = 1e-4

Filterreg
seq: source-(i+1).pcd, target-i.pcd
voxel=0.5
n=2263
data: kitti raw data 05

Filterreg 0.04
sigma2 = 0.04, w = 0.1
time: 1.4695352660923897/frame

Filterreg 0.16
sigma2 = 0.16, w = 0.1
time: 0.7287180064474909/frame

Filterreg 0.2
sigma2 = 0.2, w = 0.1
time: 0.6543910378811589/frame
