Test Result Instruction

Filterreg
number:01
source-i.pcd target-(i+1).pcd
voxel=0.005
n=2761
file:kitti_pcl_05, no need extra transformation basis

number:02
source-(i+1).pcd target-i.pcd
voxel=0.005
n=2761
file:kitti_pcl_05, no need extra transformation basis

number:03
source-i.pcd target-(i+1).pcd
voxel=0.48
n=2263
file:kitti_raw_05, need extra transformation basis

number:04
source-(i+1).pcd target-i.pcd
voxel=0.48
n=2263
file:kitti_raw_05, need extra transformation basis


CPD
number:01
source-i.pcd target-(i+1).pcd
voxel=0.4
n=2263
file:kitti_raw_05, need extra transformation basis

number:02
source-(i+1).pcd target-i.pcd
voxel=0.4
n=2263
file:kitti_raw_05, need extra transformation basis

number:03
source-i.pcd target-(i+1).pcd
voxel=0.005
n=2761
file:kitti_pcl_05, no need extra transformation basis

number:04
source-(i+1).pcd target-i.pcd
voxel=0.005
n=2761
file:kitti_pcl_05, no need extra transformation basis
