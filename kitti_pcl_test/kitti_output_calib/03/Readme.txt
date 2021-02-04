Test Result Instruction

SVR: (default: maxiter=1, tol=1.0e-3)
number:01
source-(i+1).pcd target-i.pcd
voxel=0.35
n=801
file:kitti_raw_03
use calibration matrix from kitti
use first tf from kitti groud truth
average time(seconds): 45.98331330210391/frame

CPD (default: maxiter=50, tol=0.001)
number: 01
source-(i+1).pcd target-i.pcd
voxel = 0.4
n = 801
MAX_ITER = 30
TOL = 1e-3
file: kitti_raw_03
use calibration matrix from kitti
use first tf from kitti groud truth
average time(seconds): 265.5992209549555/frame
error accumulate

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

