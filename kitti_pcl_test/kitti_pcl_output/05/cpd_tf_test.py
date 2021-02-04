import numpy as np

last_raw = [0, 0, 0, 1]
tf_basis = [[0, 1, 0, 0],
			[0, 0, 1, 0],
			[-1, 0, 0, 0],
			[0, 0, 0, 1]]
filename = 'cpd_corret_basis.txt'

def output(filename, tf):
		kitti = ''

		for i in range(12):
			if i == 11:
				kitti += str(tf[0][i])
			else:
				kitti = kitti + str(tf[0][i]) + ' '
		with open(filename, 'a+') as f:
			f.write(kitti + '\n')

with open('cpd_wrongtfbasis.txt', 'r') as f:
	lines = f.readlines()
	for line in lines:
		matrix = line.split()
		tf = list(map(float, matrix))
		#print(tf)
		tf = np.reshape(tf, (3, 4))
		#print(tf)
		tf = np.vstack((tf, last_raw))
		#print(tf)
		true_tf = np.dot(tf_basis, tf)
		true_tf = true_tf[0:3, :]
		true_tf = np.reshape(true_tf, (1, 12))
		output(filename, true_tf)
		


