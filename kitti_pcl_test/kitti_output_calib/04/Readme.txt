Test Result Instruction

Filterreg (default: maxiter=50, tol=0.001)
number:01
source-i.pcd target-(i+1).pcd
voxel = 0.4
n = 271
MAX_ITER = 30
TOL = 1e-3
file: kitti_raw_04
use calibration matrix from kitti
use first tf from kitti groud truth
average time(seconds): 0.982200708785186/frame

number:02
source-(i+1).pcd target-i.pcd
voxel = 0.4
n = 271
MAX_ITER = 30
TOL = 1e-3
file: kitti_raw_04
use calibration matrix from kitti
use first tf from kitti groud truth
average time(seconds): 0.9956029700703598/frame

number:03
source-i.pcd target-(i+1).pcd
voxel = 0.6
n = 271
MAX_ITER = 30
TOL = 1e-3
file: kitti_raw_04
use calibration matrix from kitti
use first tf from kitti groud truth
average time(seconds): 0.38829445873416263/frame

number:04
source-(i+1).pcd target-i.pcd
voxel = 0.6
n = 271
MAX_ITER = 30
TOL = 1e-3
file: kitti_raw_04
use calibration matrix from kitti
use first tf from kitti groud truth
average time(seconds): 0.40940919816661064/frame

number:05 default setting
source-i.pcd target-(i+1).pcd
voxel = 0.6
n = 271
MAX_ITER = 50
TOL = 1e-3
file: kitti_raw_04
use calibration matrix from kitti
use first tf from kitti groud truth
average time(seconds): 0.49028875208168754/frame

number:06 default setting
source-(i+1).pcd target-i.pcd
voxel = 0.6
n = 271
MAX_ITER = 50
TOL = 1e-3
file: kitti_raw_04
use calibration matrix from kitti
use first tf from kitti groud truth
average time(seconds): 0.49268594075180805/frame

filterreg_0.04
sigma2=0.04, w=0.1
average time(seconds): 2.4692256770037053/frame

filterreg_0.09
sigma2=0.09, w=0.1
average time(seconds): 1.7913971129703827/frame

filterreg_0.16
sigma2=0.16, w=0.1
average time(seconds): 1.3677800855257778/frame

filterreg_0.2
sigma2=0.2, w=0.1
average time(seconds): 1.2043327393481984/frame

CPD (default: maxiter=50, tol=0.001)
number: 01
source-(i+1).pcd target-i.pcd
voxel = 0.6
n = 271
MAX_ITER = 30
TOL = 1e-3
file: kitti_raw_04
use calibration matrix from kitti
use first tf from kitti groud truth
average time(seconds): 130.1478400867703/frame
error accumulate

ICP(open3D)
number:
source-(i+1).pcd target-i.pcd
voxel = 0.6
n = 271
MAX_ITER = 30
TOL = 1e-3
file: kitti_raw_04
use calibration matrix from kitti
use first tf from kitti groud truth
average time(seconds): 0.010355373125531952/frame   nan

ICP(pcl c++)
number:
source-(i+1).pcd target-i.pcd
voxel = 0.6
n = 271
MAX_ITER = 200
file: kitti_raw_04
use calibration matrix from kitti
use first tf from kitti groud truth
average time(seconds): 6.198840741/frame

GICP
number: 
source-(i+1).pcd target-i.pcd
voxel = 0.6
n = 271
MAX_ITER = 500
file: kitti_raw_04
use calibration matrix from kitti
use first tf from kitti groud truth
average time(seconds): 9.16862963/frame

GMMTree (default: maxiter=20, tol=1.0e-4)
number: 01
source-(i+1).pcd target-i.pcd
voxel = 0.63
n = 271
MAX_ITER = 30
TOL = 1e-3
file: kitti_raw_04
use calibration matrix from kitti
use first tf from kitti groud truth
average time(seconds): 44.13446925518883/frame

GMMTree (default setting)
number: 02
source-(i+1).pcd target-i.pcd
voxel = 0.63
n = 271
MAX_ITER = 20
TOL = 1e-4
file: kitti_raw_04
use calibration matrix from kitti
use first tf from kitti groud truth
average time(seconds): 53.090393498836974/frame

GMMReg (not work for 1 frame test)
number: (default: maxiter=1, tol=1.0e-3)
source-(i+1).pcd target-i.pcd
voxel = 0.63
n = 271
file: kitti_raw_04
use calibration matrix from kitti
use first tf from kitti groud truth
average time(seconds): 20.734595296022654/frame

SVR (default: maxiter=1, tol=1.0e-3)
number: 
source-(i+1).pcd target-i.pcd
voxel = 0.62
n = 271
MAX_ITER = 1
TOL = 1e-3
file: kitti_raw_04
use calibration matrix from kitti
use first tf from kitti groud truth
average time(seconds): 19.47836163697047/frame


