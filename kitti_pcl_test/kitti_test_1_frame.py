import numpy as np
import transformations as trans
from probreg import l2dist_regs, gmmtree, cpd, filterreg
from probreg import callbacks
import open3d as o3
import copy
from probreg import transformation as tf
from timeit import default_timer as timer

#VOXEL = 0.005
#VOXEL = 0.01
#VOXEL = 0.1
#VOXEL = 0.15
#VOXEL = 0.2
#VOXEL = 0.25
#VOXEL = 0.3
#VOXEL = 0.35
VOXEL = 0.4
#VOXEL = 0.45
#VOXEL = 0.5
#VOXEL = 0.6

MAX_ITER = 30
TOL = 1e-3  # for gmmtree: 1e-4, 1e-3 sometimes can't converge

source_filename = '/media/chenxin/我不是硬盘/kitti_dataset/sequences/05/pcd/0.pcd'
target_filename = '/media/chenxin/我不是硬盘/kitti_dataset/sequences/05/pcd/0.pcd'
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

for point in target.points:
	point[2] += 3.0

# Tr_04 = Tr_05
Tr_04 = np.matrix([[-1.857739385241e-03, -9.999659513510e-01, -8.039975204516e-03, -4.784029760483e-03],
				  [-6.481465826011e-03, 8.051860151134e-03, -9.999466081774e-01, -7.337429464231e-02],
				  [9.999773098287e-01, -1.805528627661e-03, -6.496203536139e-03, -3.339968064433e-01],
				  [0,                  0,                     0,                  1]])
Tr_03 = np.matrix([[2.347736981471e-04, -9.999441545438e-01, -1.056347781105e-02, -2.796816941295e-03],
				   [1.044940741659e-02, 1.056535364138e-02, -9.998895741176e-01, -7.510879138296e-02], 
				   [9.999453885620e-01, 1.243653783865e-04, 1.045130299567e-02, -2.721327964059e-01],
				   [0,                  0,                  0,                  1]])

tf_basis = Tr_04

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

'''
np_source = np.asarray(source.points)
s_points_after_tf_basis = mul_tf_basis(np_source)
source.points = o3.utility.Vector3dVector(s_points_after_tf_basis)

#print(source.points[100])

np_target = np.asarray(target.points)
t_points_after_tf_basis = mul_tf_basis(np_target)
target.points = o3.utility.Vector3dVector(t_points_after_tf_basis)
'''

'''
# gmmtree
print('Registration method: gmmtree')
start = timer()
tf_param, _ = gmmtree.registration_gmmtree(source, target) #, maxiter=MAX_ITER, tol=TOL, callbacks=[])
end = timer()
'''
'''
# cpd
print('Registration method: cpd')
start = timer()
tf_param, _, _ = cpd.registration_cpd(source, target, maxiter=MAX_ITER, tol=TOL)
end = timer()
'''

# filterreg
print('Registration method: filterreg')
print('init sigma = 0.4, init w = 0.1')
start = timer()
objective_type = 'pt2pt'
tf_param, _, _ = filterreg.registration_filterreg(source, target, objective_type=objective_type,
													sigma2=0.16, w=0.1) #, maxiter=MAX_ITER, tol=TOL)
end = timer()

'''
# svr
print('Registration method: svr')
start = timer()
tf_param = l2dist_regs.registration_svr(source, target)
end = timer()
'''
'''
# gmmreg
print('Registration method: gmmreg')
start = timer()
tf_param = l2dist_regs.registration_gmmreg(source, target)
end = timer()
'''
result = copy.deepcopy(source)

'''
# icp
print('Registration method: icp')
threshold = 1e-3
trans_init = np.asarray([[1,0,0,0],   
                         [0,1,0,0],   
                         [0,0,1,0],   
                         [0,0,0,1]])
start = timer()
reg_p2p = o3.registration.registration_icp(source, target, threshold, trans_init,
        o3.registration.TransformationEstimationPointToPoint(),
        o3.registration.ICPConvergenceCriteria(max_iteration = 200))
end = timer()
tf_matrix = reg_p2p.transformation

result = copy.deepcopy(source).transform(tf_matrix)
'''

t = tf_param.t
transpose_t = t.reshape((3, 1)) 
tf_matrix = np.hstack((tf_param.rot, transpose_t))

float_formatter = lambda x: "%.8f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})

print("result")
print(tf_matrix)
print("Time:", end - start)

result.points = tf_param.transform(result.points)

# draw result
source.paint_uniform_color([1, 0, 0])
target.paint_uniform_color([0, 1, 0])
result.paint_uniform_color([0, 0, 1])
o3.visualization.draw_geometries([source, target, result])
