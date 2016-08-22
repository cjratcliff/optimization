import tensorflow as tf
from constants import use_rnn, rnn_type, rnn_size

class MLP:
	def __init__(self, opt_net):
		self.batch_size = 1 # 32
		self.batches = 1000

		# Define architecture
		self.x = tf.placeholder(tf.float32, [None, 784])
		self.y_ = tf.placeholder(tf.float32, [None, 10])
		
		with tf.variable_scope("mlp"):
			self.W = tf.Variable(tf.truncated_normal(stddev=0.1, shape=[784,10]))
			self.b = tf.Variable(tf.constant(0.1, shape=[10]))
			
		y = tf.nn.softmax(tf.matmul(self.x,self.W) + self.b)
		y = tf.clip_by_value(y, 1e-10, 1.0) # Prevent log(0) in the cross-entropy calculation
		
		correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(self.y_,1))
		self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

		self.loss = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(y), reduction_indices=[1])) # Cross-entropy
		tf.scalar_summary('loss', self.loss)

		### Make these global?
		sgd_optimizer = tf.train.GradientDescentOptimizer(0.5)
		adam_optimizer = tf.train.AdamOptimizer()
		
		grad_var_pairs = sgd_optimizer.compute_gradients(self.loss)
		grad_var_pairs = [i for i in grad_var_pairs if 'mlp/' in i[1].name]
		grads,vars = zip(*grad_var_pairs)
		grads = [tf.reshape(i,(-1,1)) for i in grads]
		
		#if not grad_clip_value is None:
		#	grads = [tf.clip_by_value(g, -grad_clip_value, grad_clip_value) for g in grads]
			
		self.grads = tf.concat(0,grads)
		
		self.trainable_variables = [i for i in tf.trainable_variables() if 'mlp/' in i.name]
		
		self.sgd_train_step = sgd_optimizer.apply_gradients(grad_var_pairs)
		self.adam_train_step = adam_optimizer.apply_gradients(grad_var_pairs)
	
		##### Opt net #####
		input = tf.reshape(self.grads,[-1,1]) ### check

		#updates = opt_net.mean
		### Remove duplicate code - make into functions
		if use_rnn:
			output,_ = opt_net.cell(input, opt_net.rnn_state)
			output = tf.reshape(output,[m,rnn_size])
			self.mean = tf.matmul(output, opt_net.W1) + opt_net.b1
		else:
			updates = tf.matmul(input, opt_net.W1) + opt_net.b1
		
		# Apply updates to the parameters in the train net.
		self.opt_net_train_step = opt_net.update_params(self.trainable_variables, updates)
		
		self.init = tf.initialize_all_variables()
		
		