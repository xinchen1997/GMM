'''This file is to read pcd file and use GMMTree, SVR and FilterReg to align them and output trajectory '''

#import pcl
import sys
import numpy as np
#import transformations as trans
from probreg import gmmtree
from probreg import l2dist_regs
from probreg import filterreg
from probreg import callbacks
import utils
from timeit import default_timer as timer

import open3d as o3
from probreg import cpd

use_cuda = True
if use_cuda:
    import cupy as cp
    to_cpu = cp.asnumpy
    cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
else:
    cp = np
    to_cpu = lambda x: x

#VOXEL = 0.005
VOXEL = 0.4
#VOXEL = 0.18
#VOXEL = 0.25 # slow, stuck
#VOXEL = 0.3  # good
#VOXEL = 0.35  # good stuck
#VOXEL = 0.4
#VOXEL = 0.6

MAX_ITER = 30
TOL = 1e-3

# Tr_04 = Tr_05 = Tr_06 = Tr_07 = Tr_08 = Tr_09
Tr_04 = np.matrix([[-1.857739385241e-03, -9.999659513510e-01, -8.039975204516e-03, -4.784029760483e-03],
				      [-6.481465826011e-03, 8.051860151134e-03, -9.999466081774e-01, -7.337429464231e-02],
					  [9.999773098287e-01, -1.805528627661e-03, -6.496203536139e-03, -3.339968064433e-01],
					  [0,                  0,                     0,                  1]])

Tr_03 = np.matrix([[2.347736981471e-04, -9.999441545438e-01, -1.056347781105e-02, -2.796816941295e-03],
				   [1.044940741659e-02, 1.056535364138e-02, -9.998895741176e-01, -7.510879138296e-02], 
				   [9.999453885620e-01, 1.243653783865e-04, 1.045130299567e-02, -2.721327964059e-01],
				   [0,                  0,                  0,                  1]])

tf_basis = Tr_04

original_tf_09 = '1.000000e+00 1.197625e-11 1.704638e-10 5.551115e-17 1.197625e-11 1.000000e+00 3.562503e-10 0.000000e+00 1.704638e-10 3.562503e-10 1.000000e+00 2.220446e-16\n'
original_tf_08 = '1.000000e+00 1.197624e-11 1.704639e-10 3.214096e-14 1.197625e-11 1.000000e+00 3.562503e-10 -1.998401e-15 1.704639e-10 3.562503e-10 1.000000e+00 -4.041212e-14\n'
original_tf_07 = '1.000000e+00 1.197625e-11 1.704638e-10 5.551115e-17 1.197625e-11 1.000000e+00 3.562503e-10 0.000000e+00 1.704638e-10 3.562503e-10 1.000000e+00 2.220446e-16\n'
original_tf_06 = '1.000000e+00 1.197625e-11 1.704638e-10 0.000000e+00 1.197625e-11 1.000000e+00 3.562503e-10 -1.110223e-16 1.704638e-10 3.562503e-10 1.000000e+00 2.220446e-16\n'
original_tf_05 = '1 0 -1.81899e-10 0 9.31323e-10 1 0 0 -6.54836e-11 0 1 0\n'
original_tf_04 = '1.000000e+00 1.197625e-11 1.704638e-10 -5.551115e-17 1.197625e-11 1.000000e+00 3.562503e-10 0.000000e+00 1.704638e-10 3.562503e-10 1.000000e+00 2.220446e-16\n'
original_tf_03 = '1.000000e+00 -1.822835e-10 5.241111e-10 -5.551115e-17 -1.822835e-10 9.999999e-01 -5.072855e-10 -3.330669e-16 5.241111e-10 -5.072855e-10 9.999999e-01 2.220446e-16\n'

class Transformation:
	def __init__(self):
		'''
		# 09
		self.seq_tf = np.array([[1, 1.197625e-11, 1.704638e-10, 5.551115e-17], 
								[1.197625e-11, 1, 3.562503e-10, 0], 
								[1.704638e-10, 3.562503e-10, 1, 2.220446e-16],
								[0, 0, 0, 1]])
		'''
		# 08
		self.seq_tf = np.array([[1, 1.197624e-11, 1.704639e-10, 3.214096e-14], 
								[1.197625e-11, 1, 3.562503e-10, -1.998401e-15], 
								[1.704639e-10, 3.562503e-10, 1, -4.041212e-14],
								[0, 0, 0, 1]])
		'''
		# 07 = 09
		self.seq_tf = np.array([[1, 1.197625e-11, 1.704638e-10, 5.551115e-17], 
								[1.197625e-11, 1, 3.562503e-10, 0], 
								[1.704638e-10, 3.562503e-10, 1, 2.220446e-16],
								[0, 0, 0, 1]])
		
		# 06
		self.seq_tf = np.array([[1, 1.197625e-11, 1.704638e-10, 0], 
								[1.197625e-11, 1, 3.562503e-10, -1.110223e-16], 
								[1.704638e-10, 3.562503e-10, 1, 2.220446e-16],
								[0, 0, 0, 1]])
		
		# 05
		self.seq_tf = np.array([[1, 0, -1.81899e-10, 0], 
								[9.31323e-10, 1, 0, 0], 
								[-6.54836e-11, 0, 1, 0],
								[0, 0, 0, 1]])
		
		# 04
		self.seq_tf = np.array([[1.000000e+00, 1.197625e-11, 1.704638e-10, -5.551115e-17], 
								 [1.197625e-11, 1.000000e+00, 3.562503e-10, 0.000000e+00], 
								 [1.704638e-10, 3.562503e-10, 1.000000e+00, 2.220446e-16],
								 [0,            0,            0,            1]])
		
		# 03
		self.seq_tf = np.array([[1.000000e+00, -1.822835e-10, 5.241111e-10, -5.551115e-17], 
								[-1.822835e-10, 9.999999e-01, -5.072855e-10, -3.330669e-16],
								[5.241111e-10, -5.072855e-10, 9.999999e-01, 2.220446e-16],
								[0,            0,            0,            1]])
		'''

		self.number = 4071 #total
		self.time = []

	def collect_tf(self, tf_param):
		t = tf_param.t
		transpose_t = t.reshape((3, 1)) 
		tf_matrix = np.hstack((tf_param.rot, transpose_t))
		last_row = [0, 0, 0, 1]
		tf_matrix = np.vstack((tf_matrix, last_row))
		return tf_matrix

	def output(self, filename, tf):
		tf_matrix = tf[0:3, :]
		output = tf_matrix.flatten()
		kitti = ''

		for i in range(12):
			if i == 11:
				kitti += str(output[i])
			else:
				kitti = kitti + str(output[i]) + ' '
		with open(filename, 'a+') as f:
			f.write(kitti + '\n')

	def mul_tf_basis(self, points):
		n = len(points)
		new_points = [0, 0, 0]
		for i in range(n):
			point = points[i].reshape(-1, 1)
			new_point = np.dot(tf_basis, point)
			new_point = np.reshape(new_point, (1, 3))
			new_points = np.vstack((new_points, new_point[0]))
		return new_points[1:, :]

	def calib_velo2cam(self, points):
		n = len(points)
		new_points = [0, 0, 0]
		t = np.array([[1]])
		for i in range(n):
			point = points[i].reshape(-1, 1)  # 3x1
			point = np.row_stack((point, t))  # 4x1 
			new_point = np.dot(tf_basis, point)  # 4x1
			new_point = np.reshape(new_point, (1, 4))
			new_point = np.delete(new_point[0], 3)
			new_points = np.vstack((new_points, new_point[0]))
		return new_points[1:, :]
	
	def gmmtree_pcd(self):
		output_every_tf = 'gmmtree_every_tf.txt'
		output_seq_tf = 'gmmtree_seq_tf.txt'
		with open(output_every_tf, 'a+') as f:
			f.write(original_tf)
		f.close()
		with open(output_seq_tf, 'a+') as f:
			f.write(original_tf)
		f.close()
		for i in range(self.number - 1):
			print(i)
			#source_filename = 'kitti_raw_05/' + str(i) + '.pcd'
			#target_filename = 'kitti_raw_05/' + str(i + 1) + '.pcd'
			source_filename = '/media/chenxin/我不是硬盘/kitti_dataset/sequences/04/pcd/' + str(i+1) + '.pcd'
			target_filename = '/media/chenxin/我不是硬盘/kitti_dataset/sequences/04/pcd/' + str(i) + '.pcd'
			source = o3.io.read_point_cloud(source_filename)
			source = source.voxel_down_sample(voxel_size=VOXEL)
			target = o3.io.read_point_cloud(target_filename)
			target = target.voxel_down_sample(voxel_size=VOXEL)

			np_source = np.asarray(source.points)
			#s_points_after_tf_basis = self.mul_tf_basis(np_source)
			s_points_after_tf_basis = self.calib_velo2cam(np_source)
			source.points = o3.utility.Vector3dVector(s_points_after_tf_basis)

			np_target = np.asarray(target.points)
			#t_points_after_tf_basis = self.mul_tf_basis(np_target)
			t_points_after_tf_basis = self.calib_velo2cam(np_target)
			target.points = o3.utility.Vector3dVector(t_points_after_tf_basis)
			
			start = timer()
			tf_param, _ = gmmtree.registration_gmmtree(source, target) #, maxiter=MAX_ITER, tol=TOL)
			end = timer()
			self.time.append(end - start)
			curr_tf = self.collect_tf(tf_param)
			seq_tf = np.dot(curr_tf, self.seq_tf)
			self.seq_tf = seq_tf
			self.output(output_every_tf, curr_tf)
			self.output(output_seq_tf, seq_tf)
	
	# MAX_ITER = 1
	def svr_pcd(self):
		output_every_tf = 'svr_every_tf.txt'
		output_seq_tf = 'svr_seq_tf.txt'
		with open(output_every_tf, 'a+') as f:
			f.write(original_tf_09)
		f.close()
		with open(output_seq_tf, 'a+') as f:
			f.write(original_tf_09)
		f.close()
		for i in range(self.number - 1):
			print(i)
			source_filename = '/media/chenxin/我不是硬盘/kitti_dataset/sequences/09/pcd/' + str(i+1) + '.pcd'
			target_filename = '/media/chenxin/我不是硬盘/kitti_dataset/sequences/09/pcd/' + str(i) + '.pcd'
			source = o3.io.read_point_cloud(source_filename)
			source = source.voxel_down_sample(voxel_size=VOXEL)
			target = o3.io.read_point_cloud(target_filename)
			target = target.voxel_down_sample(voxel_size=VOXEL)

			np_source = np.asarray(source.points)
			#s_points_after_tf_basis = self.mul_tf_basis(np_source)
			s_points_after_tf_basis = self.calib_velo2cam(np_source)
			source.points = o3.utility.Vector3dVector(s_points_after_tf_basis)

			np_target = np.asarray(target.points)
			#t_points_after_tf_basis = self.mul_tf_basis(np_target)
			t_points_after_tf_basis = self.calib_velo2cam(np_target)
			target.points = o3.utility.Vector3dVector(t_points_after_tf_basis)

			start = timer()
			tf_param = l2dist_regs.registration_svr(source, target)
			end = timer()
			self.time.append(end - start)
			curr_tf = self.collect_tf(tf_param)
			seq_tf = np.dot(curr_tf, self.seq_tf)
			self.seq_tf = seq_tf
			self.output(output_every_tf, curr_tf)
			self.output(output_seq_tf, seq_tf)

	def gmmreg(self):
		output_every_tf = 'gmmreg_every_tf.txt'
		output_seq_tf = 'gmmreg_seq_tf.txt'
		with open(output_every_tf, 'a+') as f:
			f.write(original_tf)
		f.close()
		with open(output_seq_tf, 'a+') as f:
			f.write(original_tf)
		f.close()
		for i in range(self.number - 1):
			print(i)
			#source_filename = 'kitti_raw_05/' + str(i) + '.pcd'
			#target_filename = 'kitti_raw_05/' + str(i + 1) + '.pcd'
			source_filename = '/media/chenxin/我不是硬盘/kitti_dataset/sequences/04/pcd/' + str(i+1) + '.pcd'
			target_filename = '/media/chenxin/我不是硬盘/kitti_dataset/sequences/04/pcd/' + str(i) + '.pcd'
			source = o3.io.read_point_cloud(source_filename)
			source = source.voxel_down_sample(voxel_size=VOXEL)
			target = o3.io.read_point_cloud(target_filename)
			target = target.voxel_down_sample(voxel_size=VOXEL)

			np_source = np.asarray(source.points)
			#s_points_after_tf_basis = self.mul_tf_basis(np_source)
			s_points_after_tf_basis = self.calib_velo2cam(np_source)
			source.points = o3.utility.Vector3dVector(s_points_after_tf_basis)

			np_target = np.asarray(target.points)
			#t_points_after_tf_basis = self.mul_tf_basis(np_target)
			t_points_after_tf_basis = self.calib_velo2cam(np_target)
			target.points = o3.utility.Vector3dVector(t_points_after_tf_basis)

			start = timer()
			tf_param = l2dist_regs.registration_gmmreg(source, target)
			end = timer()
			self.time.append(end - start)
			curr_tf = self.collect_tf(tf_param)
			seq_tf = np.dot(curr_tf, self.seq_tf)
			self.seq_tf = seq_tf
			self.output(output_every_tf, curr_tf)
			self.output(output_seq_tf, seq_tf)
	
	def filterreg_pcd(self):
		output_every_tf = 'filterreg_every_tf.txt'
		output_seq_tf = 'filterreg_seq_tf.txt'
		with open(output_every_tf, 'a+') as f:
			f.write(original_tf_08)
		f.close()
		with open(output_seq_tf, 'a+') as f:
			f.write(original_tf_08)
		f.close()
		for i in range(self.number - 1):
			print(i)
			#source_filename = 'kitti_raw_05/' + str(i) + '.pcd'
			#target_filename = 'kitti_raw_05/' + str(i + 1) + '.pcd'
			source_filename = '/media/chenxin/我不是硬盘/kitti_dataset/sequences/08/pcd/' + str(i+1) + '.pcd'
			target_filename = '/media/chenxin/我不是硬盘/kitti_dataset/sequences/08/pcd/' + str(i) + '.pcd'
			source = o3.io.read_point_cloud(source_filename)
			source = source.voxel_down_sample(voxel_size=VOXEL)   #voxel 0.47/0.48
			target = o3.io.read_point_cloud(target_filename)
			target = target.voxel_down_sample(voxel_size=VOXEL)
			
			np_source = np.asarray(source.points)
			#s_points_after_tf_basis = self.mul_tf_basis(np_source)
			s_points_after_tf_basis = self.calib_velo2cam(np_source)
			source.points = o3.utility.Vector3dVector(s_points_after_tf_basis)

			np_target = np.asarray(target.points)
			#t_points_after_tf_basis = self.mul_tf_basis(np_target)
			t_points_after_tf_basis = self.calib_velo2cam(np_target)
			target.points = o3.utility.Vector3dVector(t_points_after_tf_basis)
			
			start = timer()
			objective_type = 'pt2pt'
			'''
			tf_param, _, _ = filterreg.registration_filterreg(source, target,
    			                                              objective_type=objective_type,
    			                                              maxiter=MAX_ITER, 
    			                                              tol=TOL,
        			                                          sigma2=None)
        	'''
        	# default: maxiter = 50, tol = 1e-3
			tf_param, _, _ = filterreg.registration_filterreg(source, target,
    			                                              objective_type=objective_type,
    			                                              sigma2=0.16,
    			                                              w=0.1)  # sigma = 0.2, w = 0.1
			end = timer()
			self.time.append(end - start)
			curr_tf = self.collect_tf(tf_param)
			seq_tf = np.dot(curr_tf, self.seq_tf)
			self.seq_tf = seq_tf
			self.output(output_every_tf, curr_tf)
			self.output(output_seq_tf, seq_tf)
	
	def cpd_rigid(self):
		output_every_tf = 'cpd_every_tf.txt'
		output_seq_tf = 'cpd_seq_tf.txt'
		with open(output_every_tf, 'a+') as f:
			f.write(original_tf_03)
		f.close()
		with open(output_seq_tf, 'a+') as f:
			f.write(original_tf_03)
		f.close()
		for i in range(self.number - 1):
			print(i)
			#source_filename = 'kitti_raw_05/' + str(i) + '.pcd'
			#target_filename = 'kitti_raw_05/' + str(i + 1) + '.pcd'
			source_filename = '/media/chenxin/我不是硬盘/kitti_dataset/sequences/03/pcd/' + str(i+1) + '.pcd'
			target_filename = '/media/chenxin/我不是硬盘/kitti_dataset/sequences/03/pcd/' + str(i) + '.pcd'
			source = o3.io.read_point_cloud(source_filename)
			source = source.voxel_down_sample(voxel_size=VOXEL)
			target = o3.io.read_point_cloud(target_filename)
			target = target.voxel_down_sample(voxel_size=VOXEL)
			
			np_source = np.asarray(source.points)
			#s_points_after_tf_basis = self.mul_tf_basis(np_source)
			s_points_after_tf_basis = self.calib_velo2cam(np_source)
			source.points = o3.utility.Vector3dVector(s_points_after_tf_basis)

			np_target = np.asarray(target.points)
			#t_points_after_tf_basis = self.mul_tf_basis(np_target)
			t_points_after_tf_basis = self.calib_velo2cam(np_target)
			target.points = o3.utility.Vector3dVector(t_points_after_tf_basis)
			
			start = timer()
			tf_param, _, _ = cpd.registration_cpd(source, target, maxiter=MAX_ITER, tol=TOL)
			end = timer()
			self.time.append(end - start)
			curr_tf = self.collect_tf(tf_param)
			seq_tf = np.dot(curr_tf, self.seq_tf)
			self.seq_tf = seq_tf
			self.output(output_every_tf, curr_tf)
			self.output(output_seq_tf, seq_tf)
		
	def icp(self):
		output_every_tf = 'icp_every_tf.txt'
		output_seq_tf = 'icp_seq_tf.txt'
		with open(output_every_tf, 'a+') as f:
			f.write(original_tf)
		f.close()
		with open(output_seq_tf, 'a+') as f:
			f.write(original_tf)
		f.close()

		threshold = 1e-3
		#trans_init = self.seq_tf
		for i in range(self.number - 1):
			print(i)
			#source_filename = 'kitti_raw_05/' + str(i) + '.pcd'
			#target_filename = 'kitti_raw_05/' + str(i + 1) + '.pcd'
			source_filename = '/media/chenxin/我不是硬盘/kitti_dataset/sequences/04/pcd/' + str(i) + '.pcd'
			target_filename = '/media/chenxin/我不是硬盘/kitti_dataset/sequences/04/pcd/' + str(i+1) + '.pcd'
			source = o3.io.read_point_cloud(source_filename)
			source = source.voxel_down_sample(voxel_size=VOXEL)
			target = o3.io.read_point_cloud(target_filename)
			target = target.voxel_down_sample(voxel_size=VOXEL)

			np_source = np.asarray(source.points)
			#s_points_after_tf_basis = self.mul_tf_basis(np_source)
			s_points_after_tf_basis = self.calib_velo2cam(np_source)
			source.points = o3.utility.Vector3dVector(s_points_after_tf_basis)

			np_target = np.asarray(target.points)
			#t_points_after_tf_basis = self.mul_tf_basis(np_target)
			t_points_after_tf_basis = self.calib_velo2cam(np_target)
			target.points = o3.utility.Vector3dVector(t_points_after_tf_basis)

			start = timer()
			reg_p2p = o3.registration.registration_icp(source, target, threshold, self.seq_tf,
        			  o3.registration.TransformationEstimationPointToPoint(),
        			  o3.registration.ICPConvergenceCriteria(max_iteration = 30))
			end = timer()
			self.time.append(end - start)

			curr_tf = reg_p2p.transformation
			seq_tf = np.dot(curr_tf, self.seq_tf)
			self.seq_tf = seq_tf
			self.output(output_every_tf, curr_tf)
			self.output(output_seq_tf, seq_tf)

def main():
	my_tf = Transformation() 
	algorithm = sys.argv[1]
	
	if algorithm == 'gmmtree':
		my_tf.gmmtree_pcd()
	
	if algorithm == 'svr':
		my_tf.svr_pcd()

	if algorithm == 'gmmreg':
		my_tf.gmmreg()
	
	if algorithm == 'filterreg':
		my_tf.filterreg_pcd()
	
	if algorithm == 'cpd':
		my_tf.cpd_rigid()

	if algorithm == 'icp':
		my_tf.icp()

	print(np.mean(my_tf.time))

if __name__ == "__main__":
	main()
