from __future__ import division, print_function, absolute_import 
import speech_data
import tensorflow as tf
import numpy as np
import pdb

learning_rate = 0.0001
training_iters = 300000  # steps
batch_size = 64

width = 20  # mfcc features
height = 80  # (max) length of utterance
num_classes = 11  # digits + space

# Network building
# (?, 20, 80)
model_input = tf.placeholder(tf.float32, shape=(batch_size, width, height))

model_target_ixs = tf.placeholder(tf.int64, name="target_ixs")
model_target_vals = tf.placeholder(tf.int32, name="target_vals")
model_target_shape = tf.placeholder(tf.int64, name="target_shape")
model_targetY = tf.SparseTensor(model_target_ixs, model_target_vals, model_target_shape)
model_seq_lengths = tf.placeholder(tf.int32, shape=(batch_size))

# (?, 128)
cell = tf.nn.rnn_cell.BasicLSTMCell(128)
cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=0.8, output_keep_prob=0.8)
rnn_input = [tf.squeeze(input, axis=1) for input in tf.split(model_input, width, axis=1)]

# rnn_input.shape = list(20) of 64X128
rnn_output, rnn_state = tf.nn.static_rnn(cell, rnn_input, dtype=tf.float32)

model_fc_w = tf.get_variable("fc_w", shape=(128, num_classes))
model_fc_b = tf.get_variable("fc_b", shape=(num_classes))
model_logits = [tf.matmul(t, model_fc_w) + model_fc_b for t in rnn_output]
model_logits3d = tf.stack(model_logits)
model_loss = tf.reduce_mean(tf.nn.ctc_loss(model_targetY, model_logits3d, model_seq_lengths))
model_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(model_loss)
model_predict = tf.to_int32(tf.nn.ctc_beam_search_decoder(model_logits3d, model_seq_lengths, merge_repeated=False)[0][0])

batch = speech_data.mfcc_batch_generator(batch_size, target=speech_data.Target.dense)
X, Y, batch_no = next(batch)
trainX, trainY = X, Y
testX, testY = X, Y #overfit for now

pdb.set_trace()

epochs = 4

session = tf.Session()
session.run(tf.global_variables_initializer())
epoch = 0
epochs = 500
while epoch < epochs:
  epoch += 1
  print("epoch {0}".format(epoch))
  num_batches = int(len(trainX) / batch_size)
  batch_no = 1  # set to get in the loop
  while batch_no > 0:

    feedDict = {model_input: trainX, 
                model.targetIxs: batchTargetIxs,
                model.targetVals: batchTargetVals, 
                model.targetShape: batchTargetShape,
                model.seqLengths: batchSeqLengths}

    #feed_dict = {model_input: trainX, model_output: trainY}
    loss, _ = session.run([model_loss, model_train], feed_dict)
    X, Y, batch_no = next(batch)
    trainX, trainY = X, Y
    testX, testY = X, Y #overfit for now

  # check it
  batch_no = 1
  right = 0
  wrong = 0
  while batch_no > 0:
    predict = session.run(model_predict, {model_input: trainX})
    for i in range(batch_size):
      if np.argmax(predict[i]) == np.argmax(trainY[i]):
        right += 1
      else:
        wrong += 1
    X, Y, batch_no = next(batch)
    trainX, trainY = X, Y
    testX, testY = X, Y #overfit for now
  
  print("right({0}) wrong({1})".format(right, wrong))

