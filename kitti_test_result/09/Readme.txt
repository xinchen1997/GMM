Test Result Instruction

SVR:
default setting
source-(i+1).pcd target-i.pcd
voxel=0.4
n=1591
file:kitti_raw_09
time: 36.31128579295385/frame
use calibration matrix from kitti

Filterreg
default setting
source-(i+1).pcd target-i.pcd
voxel=0.4
n=1591
file:kitti_09
time: 0.8346514637835885/frame

Filterreg 0.04
sigma2 = 0.04, w = 0.1
time: 1.96493161656561/frame

Filterreg 0.16
sigma2 = 0.16, w = 0.1
time: 1.0731953263038414/frame

Filterreg 0.2
sigma2 = 0.2, w = 0.1
time: 0.9670746300011652/frame
