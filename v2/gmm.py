import tensorflow as tf
import numpy as np

from constants import m, num_gaussians, cov_range, weight_gaussians
from ac_network import inv_scale_grads

class GMM(object):
	def __init__(self):
		
		self.mean_vectors = []
		self.inv_cov_matrices = []
		
		for i in range(num_gaussians):
			self.mean_vectors.append(np.random.rand(m,1))

			# Covariance matrices must be positive-definite
			Q = np.random.rand(m,m)*cov_range[1]
			Q_T = np.transpose(Q)
			
			D = np.abs(np.random.rand(m)*cov_range[1])
			D = np.diagflat(D)
			
			C = np.dot(np.dot(Q_T,D),Q)
			C = np.power(C,0.33)/m # Re-scale
			C = np.linalg.inv(C)
			self.inv_cov_matrices.append(C)
		
		
	def gmm_loss(self, points):
		###assert len(points.shape) == 3
		losses = []
		for i in range(num_gaussians):
			d = points - self.mean_vectors[i]
			d = np.reshape(d,(1,m))
			loss = np.dot(d,self.inv_cov_matrices[i])
			loss = np.square(loss)
			loss = -np.exp(-0.5*loss)
			losses.append(loss)
		return np.mean(losses)
		
		
	def gen_points(self,num_points):
		### Use points near the means?
		point = np.random.rand(m*num_points)
		point = np.reshape(point,[1,m,num_points])
		return point
		
		
	def choose_action(self,mean,variance):
		for i,v in enumerate(variance):
			mean[i] += np.random.normal(0,v)
		
		mean = inv_scale_grads(mean)	
		return mean
	
	
	def act(self, state, action):	
		action = np.reshape(action,[m])
		state.point += action
		loss = self.gmm_loss(np.reshape(state.point,[1,m,1]))
		reward = -loss ### check sign
		return reward, state
		

class State(object):
	def __init__(self,gmm):

		### Generate a point near the means?
		self.point = np.random.rand(m)
		
		### Creating a session probably takes a lot of computation
		sess = tf.Session() ### Should use the existing session - caused graph difficulties
		
		##### Graph to compute the gradients #####
		point_ = tf.placeholder(tf.float32, [m])
		mean_vectors_ = tf.placeholder(tf.float32, [num_gaussians,m,1])
		inv_cov_matrices_ = tf.placeholder(tf.float32, [num_gaussians,m,m])
		
		point = tf.reshape(point_, [1,m,1])
		point = tf.tile(point, multiples=[1,1,num_gaussians])
		mean_vectors = tf.reshape(mean_vectors_, [1,m,num_gaussians])
		d = point - mean_vectors # 1,m,num_gaussians

		losses = tf.batch_matmul(tf.transpose(d,[2,0,1]),inv_cov_matrices_)
		# Follows the code in SciPy's multivariate_normal
		losses = tf.square(losses) # element-wise (num_gaussians,1,m)
		losses = tf.reduce_sum(losses,[2]) # Sum over the dimensions (num_gaussians,1)
		
		if weight_gaussians:
			raise NotImplementedError
		
		# The pdfs of the Gaussians are negative in order to create a minimization problem.
		losses = -tf.exp(-0.5*losses)
		losses = tf.reduce_mean(losses,[0]) # Average over the Gaussians
		
		grads = tf.gradients(losses,point_)[0]
		
		init = tf.initialize_all_variables()
		sess.run(init)
		
		self.grads = sess.run([grads],feed_dict={point_:self.point, mean_vectors_:gmm.mean_vectors, inv_cov_matrices_:gmm.inv_cov_matrices})
		self.grads = self.grads[0]