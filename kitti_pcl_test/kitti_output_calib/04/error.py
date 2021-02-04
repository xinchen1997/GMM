import numpy as np
import os

groundtruth = '04.txt'
result = 'cpd_every_tf.txt'

origin = np.array([0, 0, 0, 1])  # 1x4

def calculate(filename):
	points = []
	points.append(origin)
	last_row = [0, 0, 0, 1]

	tf_matrice = np.loadtxt(filename, delimiter=' ')
	curr_p = origin
	for matrix in tf_matrice:
		curr_p = curr_p.reshape(-1, 1)  # 4x1
		#print(curr_p)
		matrix = matrix.reshape(3, 4)
		tf_matrix = np.vstack((matrix, last_row)) # 4x4
		#print(tf_matrix)
		next_p = np.dot(tf_matrix, curr_p)  # 4x1
		#print(next_p)
		next_p = next_p.reshape(1, -1) # 1x4
		#print(next_p)
		points.append(next_p[0])
		curr_p = next_p

	return points

points1 = calculate(groundtruth)
points2 = calculate(result)

# L2 distance
def error(points1, points2):
	n = len(points1)
	error = 0
	for i in range(n):
		point1 = points1[i]
		point2 = points2[i]
		d = np.sqrt(np.sum((point1 - point2)**2))
		#print(d)
		error += d

	return error

e = error(points1, points2)
print(e)
