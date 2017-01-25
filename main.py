from __future__ import division
import argparse
import random

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import numpy as np

from constants import *
from snf import SNF, State, StateOps
from optimizer import Optimizer
from mlp import SoftmaxNet

"""
python main.py -s

pyflakes main.py compare.py optimizer.py constants.py snf.py nn_utils.py mlp.py
pylint --rcfile=pylint.cfg main.py compare.py optimizer.py constants.py snf.py nn_utils.py mlp.py
"""

def main(): ### Solve speed problems - should be considerably faster than it is?
	parser = argparse.ArgumentParser()
	parser.add_argument('--save', '-s', dest='save_model', action='store_true')
	parser.set_defaults(save=False)
	args = parser.parse_args()

	if args.save_model:
		print("Model will be saved")
	else:
		print("Model will not be saved")
		
	sess = tf.Session()
	state_ops = StateOps()
	
	opt_net = Optimizer()
	
	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
	net = SoftmaxNet(opt_net)
	
	snfs = []
	# Generate the set of SNFs
	print("Generating SNFs...")
	for i in range(num_SNFs):
		snf = SNF()
		snfs.append(snf)
	
	print("Initializing replay memory...")
	replay_memory = []
	
	# Add some initial states to the replay memory
	for i in range(replay_mem_start_size):
		snf = random.choice(snfs)

		# Initializer computes a random point and the SNF loss
		state = State(snf, state_ops, sess)
		replay_memory.append(state)
	
	init = tf.global_variables_initializer()
	sess.run(init)
	
	best_loss = np.float('inf')
	best_accuracy = 0
	print("Iter.  Loss  Sign change  Counter")

	# Training loop
	for i in range(num_iterations):
		batch_losses = []
		batch_loss_change_sign = []
		batch_grads = []
		batch_counters = []
		
		for j in range(batch_size):
			# Retrieve a random starting point from the replay memory
			state = random.choice(replay_memory)
			snf = state.snf
			
			if state.counter >= episode_length:
				snf = random.choice(snfs)
				state = State(snf, state_ops, sess)
				
			batch_counters.append(state.counter)
			prev_counter = state.counter
				
			# The RNN state is initially zero but this will become
			# rarer as old states are put back into the replay memory
			
			feed_dict = {opt_net.point: state.point,
							opt_net.variances: snf.variances, 
							opt_net.weights: snf.weights, 
							opt_net.hyperplanes: snf.hyperplanes,
							opt_net.initial_rnn_state: state.rnn_state}
			
			res = sess.run([opt_net.new_point,
							opt_net.rnn_state_out,
							opt_net.loss_change_sign,
							opt_net.loss,
							opt_net.train_step]
							+ [g for g,v in opt_net.gvs], 
							feed_dict=feed_dict)
														
			new_point, rnn_state_out, loss_change_sign, loss = res[0:4]
			
			# Prepare a new state to add to the replay memory
			state = State(snf, state_ops, sess)
			state.point = new_point
			state.rnn_state = rnn_state_out
			state.counter = prev_counter + seq_length

			# Prevent these attributes from being used until their values are overridden
			state.loss = None
			state.grads = None
				
			# Only the last state is added. Adding more may result in a loss 
			# of diversity in the replay memory
			replay_memory.append(state)
			
			if len(replay_memory) > replay_memory_max_size:
				replay_memory = replay_memory[-replay_memory_max_size:]
			
			batch_losses.append(loss)
			batch_loss_change_sign.append(loss_change_sign)
		
		loss = np.mean(batch_losses)
		avg_loss_change_sign = np.mean(batch_loss_change_sign)
		avg_counter = np.mean(batch_counters)
			
		if i % summary_freq == 0 and i > 0:
			print("{:>3}{:>10.3}{:>10.3}{:>10.3}".format(i, loss, avg_loss_change_sign, avg_counter))
				
		# Save model
		if args.save_model and avg_loss_change_sign < -0.5:
			if loss < best_loss:
				best_loss = loss
				saver = tf.train.Saver(tf.trainable_variables())
				saver.save(sess, save_path)
				print("Model saved: %f" % loss)

if __name__ == "__main__":
	main()
	
	