# -*- coding: utf-8 -*-
"""
Change Point Detection with dummy dataset

This projects amounts as much to change point detection as to pattern localization.
In generate_data() we generate either one signal or a combintion of many signals.
In the second case, there's also an option to introduce a pettern at the changepoint
The LSTM will target a label, being 0 for the base class and 1 for the target class

Made by Rob Romijnders
romijndersrob@gmail.com

Made on June 3 2016


Convention dor the data formats
X is a matrix in R^{N x D}
y is a matrix in R^{N,} not to donfuse with {N,1}

@author: rob
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
from tensorflow.python.ops import clip_ops
from scipy import signal

from tensorflow.models.rnn import seq2seq
from tensorflow.models.rnn import rnn, rnn_cell

"""Hyperparameters"""
hidden_size = 60       	#hidden size of the LSTM
output_size = 2      		# For later expansion
batch_size = 50    		# batch_size
seq_len = 50			# How long do you want the vectors with integers to add be?
drop_out = 0.8 			# Drop out
num_layers = 2			# Number of RNN layers
max_iterations=1000		# Number of iterations to train with
plot_every = 100		# How often you want terminal output?
ratio = 0.8    			# Ratio for train val split
lr_rate = 0.005			# learning rate

#Filter characteristics
fs = 500.0  #sample frequency
cutoff1 = 100.0
cutoff0 = 200.0


# Create a pattern for the transition between one signal and the other
# Note that also without this pattern, we achieve good performance
pattern = np.array([0.0, 0.5, 1.0, 0.8, 0.6, 0.4, 0.2, 0.0, -0.2, -0.4, -0.6, -0.8, -1.0, -0.5, 0.0])		
pattern_len = len(pattern)

def generate_noise(D,N):
  """Generate data for the changepoint detection. Data can either be of type 0
  or type 1, but when it's a combination fo both, we define a target label
  Input
  - D,N Dimenstionality arguments D dimensions over N samples
  Output
  - Data in format
  X is a matrix in R^{N x D}
  y is a matrix in R^{N,} not to donfuse with {N,1}"""
  #Check if we have even D, so we can split the array in future
  assert D%2 == 0, 'We need even number of dimensions'
  Dhalf = int(D/2)
  ratioP = 0.5   #balance of targets
  X = np.random.randn(N,D)
  y = np.zeros(N)
  #Generate two filter cofficients
  filters = {}
  filters['b1'],filters['a1'] = signal.butter(4,2.0*cutoff1/fs,btype='lowpass')
  filters['b0'],filters['a0'] = signal.butter(4,2.0*cutoff0/fs,btype='lowpass')
  for i in xrange(N):
    if np.random.rand() > 0.5:	#Half of the samples will have changepoint, other half wont
      signalA = signal.filtfilt(filters['b1'],filters['a1'],X[i])
      signalB = signal.filtfilt(filters['b0'],filters['a0'],X[i])
      X[i] = np.concatenate((signalA[:Dhalf],signalB[:Dhalf]),axis=0)    #Concatenate the two signals
      if True:  #Boolean: do you want to introduce a pattern at the changepoint?
        Dstart = int(Dhalf-pattern_len/2)
        X[i,Dstart:Dstart+pattern_len] = pattern
      y[i] = 1		#The target label
    else:
      mode = int(np.random.rand()>ratioP)
      X[i] = signal.filtfilt(filters['b'+str(mode)],filters['a'+str(mode)],X[i])
      y[i] = 0		#The target label
  return X,y   
  
	
		
"""Set up the data"""
N = 10000		#How many point you want to generate?
data,labels = generate_noise(seq_len,N)
#Shuffle the data
ind_cut = int(ratio*N)
ind = np.random.permutation(N)
X_train = data[:ind_cut]
X_val = data[ind_cut:]
y_train = labels[:ind_cut]
y_val = labels[ind_cut:]
N = X_train.shape[0]
Nval = X_val.shape[0]
data = None  #we don;t need to store this big matrix anymore		

"""Plot some Data"""
Nplot = 5		#Numer of rows for which you want a plot
count_pos = 0
count_neg = 0
f, axarr = plt.subplots(Nplot, 2)
while count_pos < Nplot or count_neg < Nplot:
  ind = np.random.randint(N)  
  if y_train[ind] == 0 and count_neg < Nplot:
    axarr[count_neg,0].plot(X_train[ind])
    count_neg += 1
  elif y_train[ind] == 1 and count_pos < Nplot:
    axarr[count_pos,1].plot(X_train[ind])
    count_pos += 1
plt.show()

with tf.name_scope("Placeholders") as scope:
#The place holders for the model
  inputs = [tf.placeholder(tf.float32,shape=[batch_size,1]) for _ in range(seq_len)]
  target = tf.placeholder(tf.int64, shape=[batch_size], name = 'Target')
  keep_prob = tf.placeholder("float", name = 'Drop_Out_keep_probability')		
		
with tf.name_scope("Cell") as scope:
  #Define one cell, stack the cell to obtain many layers of cell and wrap a DropOut
  cell = rnn_cell.BasicLSTMCell(hidden_size)
  cell = rnn_cell.MultiRNNCell([cell] * num_layers)
  cell = rnn_cell.DropoutWrapper(cell,output_keep_prob=keep_prob)			
  initial_state = cell.zero_state(batch_size, tf.float32)


with tf.name_scope("RNN") as scope:
  # Thanks to Tensorflow, the entire decoder is just one line of code:
  outputs, states = seq2seq.rnn_decoder(inputs, initial_state, cell)
  final = outputs[-1]

with tf.name_scope("Output") as scope:
  #Map the final output state to out distribution
  W_o = tf.Variable(tf.random_normal([hidden_size,output_size], stddev=0.01))     
  b_o = tf.Variable(tf.random_normal([output_size], stddev=0.01))
  prediction = tf.matmul(final, W_o) + b_o

with tf.name_scope("Optimization") as scope:
  #Optimize with cross entropy error between the true distribution and the 
  # distribution following the SoftMax
  cost = tf.nn.sparse_softmax_cross_entropy_with_logits(prediction, target)
  train_op = tf.train.RMSPropOptimizer(lr_rate, 0.2).minimize(cost)
  loss = tf.reduce_sum(cost)
				
				
with tf.name_scope("Evaluating_accuracy") as scope:
  correct_prediction = tf.equal(tf.argmax(prediction,1), target)
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

#Fire up session
sess = tf.Session()
sess.run(tf.initialize_all_variables())

#Collect information
# For now, we take old-fashioned numpy approach. In future, we might
# use TensorBoard for this part
perf_collect = np.zeros((int(np.floor(max_iterations /plot_every)),4))
step = 0
for k in range(1,max_iterations):
    batch_ind = np.random.choice(N,batch_size,replace=False)
    X = X_train[batch_ind]
    y = y_train[batch_ind]
    X = np.split(X,seq_len,axis=1)


    #Create the dictionary of inputs to feed into sess.run
    train_dict = {inputs[i]:X[i] for i in range(seq_len)}
    train_dict.update({target: y, keep_prob:drop_out})

    result = sess.run([train_op,loss, accuracy],feed_dict=train_dict)   #perform an update on the parameters
    cost_train = result[1]
    acc_train = result[2]

    
    if (k%plot_every==0):   #Output information
        
        batch_ind_sub = np.random.choice(Nval,batch_size,replace=False)
        X_val_sub = X_val[batch_ind_sub]
        y_val_sub = y_val[batch_ind_sub]
		
        X_val_sub = np.split(X_val_sub,seq_len,axis=1)				
        val_dict = {inputs[i]:X_val_sub[i] for i in range(seq_len)}  #create validation dictionary
        val_dict.update({target: y_val_sub, keep_prob:1.0})
        result = sess.run([loss,accuracy],feed_dict = val_dict )            #compute the cost on the validation set
        cost_val = result[0]
        acc_val = result[1]
        perf_collect[step,0] = cost_train
        perf_collect[step,1] = cost_val
        perf_collect[step,2] = acc_train
        perf_collect[step,3] = acc_val
								
        
        print('At %.0f/%.0f Cost %.4f/%.4f Accuracy %.3f/%.3f'%(k,max_iterations,cost_train,cost_val,acc_train,acc_val))
        step += 1
result = sess.run([prediction],feed_dict=val_dict)


plt.plot(perf_collect[:,0],label='train cost')
plt.plot(perf_collect[:,1],label='val cost')
plt.legend()
plt.show()
