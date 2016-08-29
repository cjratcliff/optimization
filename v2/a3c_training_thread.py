from __future__ import division

import tensorflow as tf
import numpy as np
import random

from accum_trainer import AccumTrainer
from ac_network import A3CRNN, A3CFF
from snf import SNF, State, StateOps
from constants import local_t_max, entropy_beta, use_rnn, m, discount_rate, \
						termination_prob, max_time_steps, lr_high, lr_low


class A3CTrainingthread(object):
	def __init__(self,
			 thread_index,
			 global_network,
			 learning_rate_input,
			 grad_applier,
			 num_trainable_vars,
			 snf):
			 
		# All ops to be executed in a thread must be defined here since tf.Graph is not thread-safe.
		
		self.thread_index = thread_index
		self.learning_rate_input = learning_rate_input
		#self.max_global_time_step = max_global_time_step
		self.snf = snf
		self.state_ops = StateOps()
		self.episode_reward = 0
		
		if use_rnn:
			initializer = tf.random_uniform_initializer(-0.1, 0.1)		
			with tf.variable_scope("model"+str(thread_index), reuse=None, initializer=initializer):
				self.local_network = A3CRNN(num_trainable_vars)
		else:
			self.local_network = A3CFF(num_trainable_vars)
			
		self.local_network.prepare_loss(entropy_beta)

		self.trainer = AccumTrainer()
		self.trainer.prepare_minimize(self.local_network.total_loss, self.local_network.trainable_vars)

		self.accum_gradients = self.trainer.accumulate_gradients()
		self.reset_gradients = self.trainer.reset_gradients()
		self.apply_gradients = grad_applier.apply_gradients(
			global_network.trainable_vars,
			self.trainer.get_accum_grad_list() )

		self.sync = self.local_network.sync_from(global_network)
		self.local_t = 0
		self.W = global_network.W1


	def _anneal_learning_rate(self, global_time_step):
		t = global_time_step / max_time_steps # Proportion of total time elapsed
		lr = lr_high*t + (1-t)*lr_low
		return lr


	# Run for one episode
	def thread(self, sess, global_t):
		states = []
		actions = []
		rewards = []
		values = []#[1.0] # 1.0 is the value of the first state - causes the weights to increase in FF - why?
		
		terminal_end = False
		
		if use_rnn:
			self.local_network.reset_rnn_state(1,m)
			
		# reset accumulated gradients
		sess.run(self.reset_gradients)
		
		# copy weights from shared to local
		sess.run(self.sync)
		
		start_local_t = self.local_t
		
		self.snf = SNF() # Generate a new landscape
		state = State(self.snf, self.state_ops, sess) # Generate a new starting point on the landscape

		discounted_reward = 0
		value_ = 1.0
		
		for i in range(local_t_max):
			if use_rnn:
				mean,variance = self.local_network.run_policy(sess, state.grads, update_rnn_state=True)
			else:
				mean,variance = self.local_network.run_policy(sess, state.grads)
			
			action = self.snf.choose_action(mean,variance) # Calculate update
			states.append(state)
			actions.append(action)

			# Calculate the value of next_state
			if use_rnn:
				v = self.local_network.run_value(sess, state.grads, action, update_rnn_state=False)
			else:
				v = self.local_network.run_value(sess, state.grads, action)

			value_ *= v
			values.append(value_)

			# State is the point, action is the update
			reward, next_state = self.snf.act(state, action, self.state_ops, sess)
			
			self.episode_reward += reward
			rewards.append(reward)

			self.local_t += 1
			state = next_state

			terminal = random.random() < termination_prob
				
			if terminal: 
				terminal_end = True
				discounted_reward = (discount_rate**i)*self.episode_reward
				self.episode_reward = 0
				state = State(self.snf,self.state_ops,sess)
				if use_rnn:
					self.local_network.reset_rnn_state(1,m)
				break

		R = 0.0
		#if not terminal_end:
			# Remove the last value
			#values = values[:-1] 
		#	R = values[-1]

		# Order from the final time point to the first
		actions.reverse()
		states.reverse()
		rewards.reverse()
		values.reverse()
		
		# compute and accumulate gradients
		for (a, r, state, V) in zip(actions, rewards, states, values):
			R = r + discount_rate * R
			td = R - V # temporal difference
			
			# grads is the state, here
			sess.run(self.accum_gradients,
								feed_dict = {
									self.local_network.grads: state.grads,
									self.local_network.update: a,
									self.local_network.a: a,
									self.local_network.td: [td],
									self.local_network.r: [R]})
			 
		cur_learning_rate = self._anneal_learning_rate(global_t)

		sess.run(self.apply_gradients, feed_dict = {self.learning_rate_input: cur_learning_rate})

		diff_local_t = self.local_t - start_local_t
		return diff_local_t, discounted_reward
		
