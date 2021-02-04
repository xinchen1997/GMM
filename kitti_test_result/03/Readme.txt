*_every_tf is the transformation for every two frames
*_seq_tf is the sequential transformation

SVR: (default: maxiter=1, tol=1.0e-3)
seq: source-(i+1).pcd, target-i.pcd
voxel=0.35
number of frames: 801
data: kitti raw data 03
average run time(seconds): 45.98331330210391/frame

CPD (default: maxiter=50, tol=0.001)
seq: source-(i+1).pcd, target-i.pcd
voxel = 0.4
n = 801
MAX_ITER = 30
TOL = 1e-3
data: kitti raw data 03
average time(seconds): 265.5992209549555/frame

Filterreg
default setting
source-(i+1).pcd target-i.pcd
voxel=0.4
n=801
file:kitti_03
time: 0.8561572007110589/frame

Filterreg 0.04
sigma2 = 0.04, w = 0.1
time: 1.951403445414926/frame

Filterreg 0.16
sigma2 = 0.16, w = 0.1
time: 1.0770153553175077/frame

Filterreg 0.2
sigma2 = 0.2, w = 0.1
time: 0.9435294853788583/frame
