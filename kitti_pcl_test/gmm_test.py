import numpy as np
import open3d as o3

from gmm import GMM_CPU, GMM_Sklearn, GMM_GPU
from gmm_impl import predict
from probreg import gmmtree
import cupy
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as colors
from matplotlib import cm

#voxel = 2.5
#voxel = 0.7
VOXEL = 0.3
#voxel = 0.18
#voxel = 0.09

MAX_ITER = 50
TOL = 1e-4
COV_TYPE = 'spherical'
#COV_TYPE = 'diag'
NUM_COMPONENTS = 50
all_colors = np.array([np.random.uniform(0, 1, 3) for _ in range(NUM_COMPONENTS)])

np.random.seed(5)

source_filename = 'kitti_pcl_05/0.pcd'
target_filename = 'kitti_pcl_05/1.pcd'
source = o3.io.read_point_cloud(source_filename)
source = source.voxel_down_sample(voxel_size=VOXEL)
target = o3.io.read_point_cloud(target_filename)
target = target.voxel_down_sample(voxel_size=VOXEL)

source_np = np.asarray(source.points)
target_np = np.asarray(target.points)

gmm = GMM_GPU(n_gmm_components=NUM_COMPONENTS, max_iter=MAX_ITER, tol=TOL, cov_type=COV_TYPE)
gmm.init()

gmm.compute(source_np)

print('mean:', gmm._clf.means_)
print('covariances:', gmm._clf.covariances_)
print('weights:', gmm._clf.weights_)

gmm_idxs = gmm.predict(source_np)
print('gmm_idxs:', gmm_idxs, gmm_idxs.shape)

gmm_idxs = cupy.asnumpy(gmm_idxs)

'''
source.paint_uniform_color([1, 0, 0])
target.paint_uniform_color([0, 1, 0])
o3.visualization.draw_geometries([source, target])
'''


# Visualize
def draw_ellipse(position, covariance, component, ax=None, **kwargs):
    # gca = 'get current axis'
    #ax = ax or plt.gca()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("x axis")
    ax.set_ylabel("y axis")
    ax.set_zlabel("z axis")
    
    # turn covariance into axis
    #print(covariance.dtype)
    c = float(covariance)
    diag = [c] * 3
    #print(diag)
    cov = np.diag(diag)
    #print(cov)
    #if covariance.shape == (3, 3):
    U, s, Vt = np.linalg.svd(cov)
    radii = 1.0/np.sqrt(s)

    u = np.linspace(0.0, 2.0 * np.pi, 60)
    v = np.linspace(0.0, np.pi, 60)
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))

    #set colour map so each ellipsoid as a unique colour
    norm = colors.Normalize(vmin=0, vmax=NUM_COMPONENTS)
    cmap = cm.jet
    m = cm.ScalarMappable(norm=norm, cmap=cmap)

    ax.plot_surface(x, y, z,  rstride=3, cstride=3, color=m.to_rgba(component), linewidth=0.1, alpha=0.1, shade=True)
    plt.show()
    #ax.plot_surface(x, y, z,  rstride=3, cstride=3, color='g', linewidth=0.1, alpha=0.1, shade=True)

def plot_gmm(gmm, X, gmm_idxs, ax=None):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.set_xlabel("x axis")
	ax.set_ylabel("y axis")
	ax.set_zlabel("z axis")

	labels = gmm_idxs
	#print('X', X)
	ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels, s=20, cmap='viridis')
	plt.show()

	w_factor = 0.2 / gmm.weights_.max()  # ????
	for i, (pos, covar, w) in enumerate(zip(gmm.means_, gmm.covariances_, gmm.weights_)):
		draw_ellipse(pos, covar, i, alpha=w * w_factor)

	#plt.show()

plot_gmm(gmm._clf, source_np, gmm_idxs)