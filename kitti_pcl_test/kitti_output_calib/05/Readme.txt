Test Result Instruction

MAX_ITER = 30
TOL = 1e-4

Filterreg
default setting
source-(i+1).pcd target-i.pcd
voxel=0.4
n=2761
file:kitti_raw_05
time: 0.5726103252775552/frame
use calibration matrix from kitti

Filterreg 0.04
sigma2 = 0.04, w = 0.1
time: 1.4695352660923897/frame

Filterreg 0.16
sigma2 = 0.16, w = 0.1
time: 0.7287180064474909/frame

Filterreg 0.2
sigma2 = 0.2, w = 0.1
time: 0.6543910378811589/frame

CPD
number:01
source-i.pcd target-(i+1).pcd
voxel=0.4
n=2263
file:kitti_raw_05
use calibration matrix from kitti

number:02
source-(i+1).pcd target-i.pcd
voxel=0.4
n=2263
file:kitti_raw_05
use calibration matrix from kitti

SVR:
number:01
source-(i+1).pcd target-i.pcd
voxel=0.4
n=2761
file:kitti_raw_05
time: 30.724412275305223/frame
use calibration matrix from kitti
