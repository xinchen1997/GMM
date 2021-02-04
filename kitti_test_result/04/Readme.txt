*_every_tf is the transformation for every two frames
*_seq_tf is the sequential transformation

CPD
seq: source-(i+1).pcd, target-i.pcd
voxel = 0.6
number of frames: 271
MAX_ITER = 30
TOL = 1e-3
data: kitti raw data 04
average time(seconds): 130.1478400867703/frame

SVR (default: maxiter=1, tol=1.0e-3)
seq: source-(i+1).pcd, target-i.pcd
voxel = 0.62
number of frames: 271
data: kitti raw data 04
average run time(seconds): 19.47836163697047/frame

Filterreg
seq: source-(i+1).pcd, target-i.pcd
voxel = 0.6
number of frames = 271
MAX_ITER = 50
TOL = 1e-3
data: kitti raw data 04
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

ICP (use pcl c++)
seq: source-(i+1).pcd, target-i.pcd
voxel = 0.6
number of frames = 271
MAX_ITER = 200
data: kitti raw data 04
average time(seconds): 6.198840741/frame

GICP
seq: source-(i+1).pcd, target-i.pcd
voxel = 0.6
number of frames = 271
MAX_ITER = 500
data: kitti raw data 04
average time(seconds): 9.16862963/frame

GMMTree
seq: source-(i+1).pcd, target-i.pcd
voxel = 0.63
number of frames = 271
MAX_ITER = 20
TOL = 1e-4
data: kitti raw data 04
average time(seconds): 53.090393498836974/frame

GMMReg (not work for 1 transformation test)
seq: source-(i+1).pcd, target-i.pcd
voxel = 0.63
n = 271
maxiter=1
tol=1.0e-3
data: kitti raw data 04
average time(seconds): 20.734595296022654/frame



