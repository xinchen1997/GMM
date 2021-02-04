import numpy as np
from numpy.testing import assert_equal

import pcl
# from pcl.pcl_registration import icp, gicp, icp_nl
from pcl import IterativeClosestPoint, GeneralizedIterativeClosestPoint, IterativeClosestPointNonLinear
from pcl import GeneralizedIterativeClosestPoint

#VOXEL = 0.005
#VOXEL = 0.01
#VOXEL = 0.1
#VOXEL = 0.15
#VOXEL = 0.2
#VOXEL = 0.25
VOXEL = 0.4
#VOXEL = 0.35
#VOXEL = 0.4
#VOXEL = 0.45
#VOXEL = 0.5
#VOXEL = 0.6

MAX_ITER = 30
TOL = 1e-4  # for gmmtree: 1e-4, 1e-3 sometimes can't converge

#source_filename = 'kitti_pcl_05/0.pcd'
#target_filename = 'kitti_pcl_05/1.pcd'
source_filename = 'kitti_raw_05/0.pcd'
target_filename = 'kitti_raw_05/1.pcd'
source = o3.io.read_point_cloud(source_filename)
source = source.voxel_down_sample(voxel_size=VOXEL)
target = o3.io.read_point_cloud(target_filename)
target = target.voxel_down_sample(voxel_size=VOXEL)

print('source:', source)
print('target:', target)
'''
rx = np.matrix([[1, 0, 0, 0], 
	            [0, 1, 0, 0], 
	            [0, 0, 1, 0], 
	            [0, 0, 0, 1]])
ry = np.matrix([[0, 0, -1, 0], 
	            [0, 1, 0, 0], 
	            [1, 0, 0, 0], 
	            [0, 0, 0, 1]])
rz = np.matrix([[0, -1, 0, 0], 
	            [1, 0, 0, 0], 
	            [0, 0, 1, 0], 
	            [0, 0, 0, 1]])

tf_basis = np.dot(np.dot(rz, ry), rx)   #rz * ry * rx;
tf_basis = tf_batf_basis = tf_basis[0:3, 0:3]
'''

P0 = np.matrix([[7.070912000000e+02, 0.000000000000e+00, 6.018873000000e+02, 0.000000000000e+00], 
				[0.000000000000e+00, 7.070912000000e+02, 1.831104000000e+02, 0.000000000000e+00], 
				[0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 0.000000000000e+00],
				[0,                  0,                  0,                  1]])
Tr = np.matrix([[-1.857739385241e-03, -9.999659513510e-01, -8.039975204516e-03, -4.784029760483e-03],
				[-6.481465826011e-03, 8.051860151134e-03, -9.999466081774e-01, -7.337429464231e-02],
				[9.999773098287e-01, -1.805528627661e-03, -6.496203536139e-03, -3.339968064433e-01],
				[0,                  0,                     0,                  1]])
#tf_basis = np.dot(P0, Tr)
tf_basis = Tr

R_rect = np.matrix([[9.999239000000e-01, 9.837760000000e-03, -7.445048000000e-03], 
					[-9.869795000000e-03, 9.999421000000e-01, -4.278459000000e-03], 
					[7.402527000000e-03, 4.351614000000e-03, 9.999631000000e-01]])
Tr_velo_cam = np.matrix([[7.533745000000e-03, -9.999714000000e-01, -6.166020000000e-04, -4.069766000000e-03],
						 [1.480249000000e-02, 7.280733000000e-04, -9.998902000000e-01, -7.631618000000e-02], 
						 [9.998621000000e-01, 7.523790000000e-03, 1.480755000000e-02, -2.717806000000e-01],
						 [0,                  0,                  0,                   1]])  

#tf_basis = Tr_velo_cam
#print(tf_basis)

def mul_tf_basis(points):
		n = len(points)
		new_points = [0, 0, 0]
		t = np.array([[1]])
		for i in range(n):
			point = points[i].reshape(-1, 1)  # 3x1
			#print(point)
			point = np.row_stack((point, t))  # 4x1 
			#print(point)
			new_point = np.dot(tf_basis, point)  # 4x1
			#print(new_point)
			new_point = np.reshape(new_point, (1, 4))
			#print(new_point)
			new_point = np.delete(new_point[0], 3)
			#print(new_point)
			new_points = np.vstack((new_points, new_point[0]))
			#exit(0)
		return new_points[1:, :]

#print(source.points[100])

np_source = np.asarray(source.points)
s_points_after_tf_basis = mul_tf_basis(np_source)
source = pcl.PointCloud(s_points_after_tf_basis.astype(np.float32))

#print(source.points[100])

np_target = np.asarray(target.points)
t_points_after_tf_basis = mul_tf_basis(np_target)
target = pcl.PointCloud(t_points_after_tf_basis.astype(np.float32))

print('Registration method: gicp')
gicp = source.make_GeneralizedIterativeClosestPoint()
converged, transf, estimate, fitness = gicp.gicp(source, target, max_iter=1000)

print("Converged: ", converged, "Estimate: ", estimate, "Fitness: ", fitness)
print("Rotation: ")
print(transf[0:3,0:3])
print("Translation: ", transf[3, 0:3])