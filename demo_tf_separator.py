'''
  take tf_demo_strided and train for a separator. That will be shift 
  the input by 30% either way and recognize that as separator
'''

from __future__ import division, print_function, absolute_import 
import speech_data
import tensorflow as tf
import numpy as np
from collections import OrderedDict
import os.path;
from tensorflow.python.framework import graph_util
import pdb

learning_rate = 0.0001
training_iters = 300000  # steps
batch_size = 64

width = 20  # mfcc features
height = 80  # (max) length of utterance
nclasses = 11  # digits + separator

# Network building
# (?, 20, 80)
model_input = tf.placeholder(tf.float32, shape=(batch_size, width, height), name="model_input")
model_output = tf.placeholder(tf.float32, shape=(batch_size, nclasses), name="model_output")
# net = tflearn.input_data([None, width, height])

# (?, 128)
cell_size = 128
cell = tf.nn.rnn_cell.BasicLSTMCell(cell_size)
cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=0.8, output_keep_prob=0.8)
rnn_input = [tf.squeeze(input, axis=2) for input in tf.split(model_input, height, axis=2)]

# rnn_input.shape = list(80) of 64X20
rnn_output, rnn_state = tf.nn.static_rnn(cell, rnn_input, dtype=tf.float32)
# rnn_output.shape =  list of 80 of 64X128
rnn_output = tf.concat(rnn_output, 1)
# net = tflearn.lstm(net, 128, dropout=0.8)
model_fc_w = tf.get_variable("fc_w", shape=(height*128, nclasses))
model_fc_b = tf.get_variable("fc_b", shape=(nclasses))
model_logits = tf.matmul(rnn_output, model_fc_w) + model_fc_b

smodel_input = tf.placeholder(tf.float32, shape=(batch_size, width, 2*height))
srnn_input = [tf.squeeze(input, axis=2) for input in tf.split(smodel_input, 2*height, axis=2)]
# srnn_output 160X64X128
srnn_output, srnn_state = tf.nn.static_rnn(cell, srnn_input, dtype=tf.float32)
# srnn_output 64*20480
srnn_output = tf.concat(srnn_output, 1)

feature_width =  model_fc_w.get_shape()[0].value
input_width = feature_width*2
number_of_strides = height
stride_step = cell_size

slices = [tf.slice(srnn_output, [0,stride*stride_step], [batch_size, feature_width]) for stride in range(number_of_strides)]

smodel_logitss = [tf.matmul(slice, model_fc_w) + model_fc_b for slice in slices]
smodel_predicts = [tf.nn.softmax(smodel_logits) for smodel_logits in smodel_logitss]

# remove the probability of the separator from the other digits
remove_separator = False
# remove the probability of the other outputs from the current output
remove_all_others = True
if remove_separator:
  smodel_predicts_remove_separator = []
  for predicts in smodel_predicts:
    digits = tf.split(predicts, nclasses, axis=1)
    separator = digits[-1]
    remove_separator = [digit-separator for digit in digits]
    smodel_predicts_remove_separator.append(remove_separator)
  smodel_predicts_remove_separator = [ tf.concat(predicts, 1) for predicts in smodel_predicts_remove_separator ]
  smodel_predicts = smodel_predicts_remove_separator 

if remove_all_others:
  smodel_predicts_remove_others = []
  for predicts in smodel_predicts:
    digits = tf.split(predicts, nclasses, axis=1)
    remove_others = []
    for i in range(len(digits)):
      others = digits[:i] + digits[i+1:]
      remove_others.append(digits[i] - tf.add_n(others))
    smodel_predicts_remove_others.append(remove_others)
  smodel_predicts_remove_others = [ tf.concat(predicts, 1) for predicts in smodel_predicts_remove_others ]
  smodel_predicts = smodel_predicts_remove_others

model_loss = tf.losses.softmax_cross_entropy(model_output, model_logits)
opt = tf.train.AdamOptimizer(learning_rate)
model_train = opt.minimize(model_loss)

batch = speech_data.mfcc_batch_generator(batch_size, generate_separator = True)
#batch = speech_data.mfcc_sequence_batch_generator(batch_size, target=speech_data.Target.dense)
X, Y, batch_no = next(batch)
trainX, trainY = X, Y
testX, testY = X, Y #overfit for now

session = tf.Session()
session.run(tf.global_variables_initializer())
epoch = 0
epochs = 20
while epoch < epochs:
  epoch += 1
  print("epoch {0}".format(epoch))
  num_batches = int(len(trainX) / batch_size)
  batch_no = 1  # set to get in the loop
  while batch_no > 0:
    loss, _ = session.run([model_loss, model_train], {model_input: trainX, model_output: trainY})
    X, Y, batch_no = next(batch)
    trainX, trainY = X, Y
    testX, testY = X, Y #overfit for now

  # check it
  batch_no = 1
  original_right = 0
  both_right = 0
  wrong = 0
  
  def p2s(prob):
    return str(prob)[::-1].zfill(4)[::-1]

  GOOD_CUTOFF = 0.00

  def good_solution(predicts_softmax):
    return any( predict > GOOD_CUTOFF for predict in predicts_softmax )

  right = 0
  wrong = 0
  while batch_no > 0:
    train_X2 = np.zeros((64,20,160),dtype=float)
    train_X2[:,:,0:80] = trainX
    train_X2[:,:,80:160] = trainX
    # predicts list of 80 of (64,11) is list of timestamps of (batch_size, number_of_classes)
    predicts = session.run(smodel_predicts, {smodel_input: train_X2})
    for i in range(batch_size):
      digit1 = np.argmax(trainY[i])
      digit2 = np.argmax(trainY[i])
      if digit1 == 10 or digit2 == 10:
        continue
      #predicts = [predict for predict in predicts if good_solution(predict[i])]

      good_predicts = []
      for predict_at_time in predicts:
        if good_solution(predict_at_time[i]):
          good_predicts.append(predict_at_time[i])
      predictions = [np.argmax(predict_at_time) for predict_at_time in good_predicts]
      #predictions = [np.argmax(predict_at_time[i]) for predict_at_time in predicts]
      no_dups = []
      last = None
      for predict in predictions:
        if predict != last and predict != 10:
          no_dups.append(predict)
        last = predict

      if [digit1, digit2] == no_dups:
        both_right += 1
      elif len(no_dups) > 0 and digit1 == no_dups[0]:
        original_right += 1
      else:
        wrong += 1

      print("{0}{1}: {2}".format(digit1, digit2, predictions))
      print("label({0})".format(no_dups))
      [ print("\t{0}".format([p2s(round(p,2)) for p in predict[i]])) for predict in predicts ]
      #[ print("\t{0}".format([p2s(round(p,2)) if p > GOOD_CUTOFF else "    " for p in predict[i]])) for predict in predicts ]
    X, Y, batch_no = next(batch)
    trainX, trainY = X, Y
    testX, testY = X, Y #overfit for now
    
  print("right({0}) wrong({1}) both_right({2}) original_right({3})".format(both_right+original_right, wrong, both_right, original_right))
