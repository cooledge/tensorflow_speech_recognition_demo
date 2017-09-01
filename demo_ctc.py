from __future__ import division, print_function, absolute_import 
import speech_data
import tensorflow as tf
import numpy as np
import pdb

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
model_seq_lengths = tf.placeholder(tf.int32, shape=(batch_size), name="seq_lengths")

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

#learning_rate = 0.0001
#learning_rate = 0.001
#model_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(model_loss)

optimizer = tf.train.AdamOptimizer(1e-3)
gradients, variables = zip(*optimizer.compute_gradients(model_loss))
gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
model_optimizer = optimizer.apply_gradients(zip(gradients, variables))

#learning_rate = 0.001
#model_optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(model_loss)

model_predict = tf.to_int32(tf.nn.ctc_beam_search_decoder(model_logits3d, model_seq_lengths, merge_repeated=False)[0][0])
model_predict_dense = tf.sparse_to_dense(model_predict.indices, model_predict.dense_shape, model_predict.values)

batch = speech_data.mfcc_batch_generator(batch_size, target=speech_data.Target.dense)
X, Y, batch_no = next(batch)
trainX, trainY = X, Y
testX, testY = X, Y #overfit for now

epochs = 4

session = tf.Session()
session.run(tf.global_variables_initializer())

def dense_to_sparse(dense):
  idx = []
  vals = []
  shape = np.array(dense).shape
  lens = []
  for x in np.ndenumerate(dense):
    idx.append(x[0])
    vals.append(x[1])
    lens.append(1)
  return [idx, vals, shape, lens]

batchTargetIxs, batchTargetVals, batchTargetShape, batchSeqLengths = dense_to_sparse(trainY)

epoch = 0
epochs = 100
while epoch < epochs:
  epoch += 1
  print("epoch {0}".format(epoch))
  num_batches = int(len(trainX) / batch_size)
  batch_no = 1  # set to get in the loop
  while batch_no > 0:

    feed_dict = {
                 model_input: trainX, 
                 model_target_ixs: batchTargetIxs,
                 model_target_vals: batchTargetVals, 
                 model_target_shape: batchTargetShape,
                 model_seq_lengths: batchSeqLengths
                }

    loss, dense, _ = session.run([model_loss, tf.sparse_tensor_to_dense(model_targetY), model_optimizer], feed_dict)
    X, Y, batch_no = next(batch)
    trainX, trainY = X, Y
    testX, testY = X, Y #overfit for now

  # check it
  batch_no = 1
  right = 0
  wrong = 0
  while batch_no > 0:
    feed_dict = {
                 model_input: trainX, 
                 model_seq_lengths: batchSeqLengths
                }
    predict = session.run(model_predict_dense, feed_dict)
    for i in range(batch_size):
      if predict[i][0] == trainY[i][0]:
        right += 1
      else:
        wrong += 1
    X, Y, batch_no = next(batch)
    trainX, trainY = X, Y
    testX, testY = X, Y #overfit for now
  
  print("right({0}) wrong({1})".format(right, wrong))

