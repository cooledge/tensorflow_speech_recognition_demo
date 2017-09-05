'''
Train on single digits but run with a model that can recognize sequences of digits.
The same idea as the hierachy training
'''

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
predict_height = 2* 80  # (max) length of utterance
classes = 10  # digits

# Network building
# (?, 20, 80)
model_input = tf.placeholder(tf.float32, shape=(batch_size, width, height))
model_output = tf.placeholder(tf.float32, shape=(batch_size, 10))

# (?, 128)
cell = tf.nn.rnn_cell.BasicLSTMCell(128)
cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=0.8, output_keep_prob=0.8)
rnn_input = [tf.squeeze(input, axis=1) for input in tf.split(model_input, width, axis=1)]

# rnn_input.shape = list(20) of 64X128
rnn_output, rnn_state = tf.nn.static_rnn(cell, rnn_input, dtype=tf.float32)
# rnn_output.shape = 64X20, 128
rnn_output = tf.concat(rnn_output, 1)

model_fc_w = tf.get_variable("fc_w", shape=(width*128, 10))
model_fc_b = tf.get_variable("fc_b", shape=(10))
model_logits = tf.matmul(rnn_output, model_fc_w) + model_fc_b
# (? 10)
#net = tflearn.fully_connected(net, classes, activation='softmax')

model_predict = tf.nn.softmax(model_logits)
model_loss = tf.losses.softmax_cross_entropy(model_output, model_logits)
opt = tf.train.AdamOptimizer(learning_rate)
model_train = opt.minimize(model_loss)
#net = tflearn.regression(net, optimizer='adam', learning_rate=learning_rate, loss='categorical_crossentropy')

# model for prediction. reuses cell + fc layer

pdb.set_trace()
model_predict_input = tf.placeholder(tf.float32, shape=(batch_size, width, predict_height))
rnn_predict_input = [tf.squeeze(input, axis=1) for input in tf.split(model_predict_input, width, axis=1)]
rnn_predict_output, rnn_predict_state = tf.nn.static_rnn(cell, rnn_predict_input, dtype=tf.float32)
rnn_predict_output = tf.concat(rnn_predict_output, 1)
model_predict_logits = tf.matmul(rnn_predict_output, model_fc_w) + model_fc_b
model_predict_predict = tf.nn.softmax(model_predict_logits)

# Training

### add this "fix" for tensorflow version errors
'''
col = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
for x in col:
    tf.add_to_collection(tf.GraphKeys.VARIABLES, x ) 
'''

batch = speech_data.mfcc_batch_generator(batch_size)
X, Y, batch_no = next(batch)
trainX, trainY = X, Y
testX, testY = X, Y #overfit for now

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
    #print("batch_no({0})".format(batch_no))
    loss, _ = session.run([model_loss, model_train], {model_input: trainX, model_output: trainY})
    X, Y, batch_no = next(batch)
    trainX, trainY = X, Y
    testX, testY = X, Y #overfit for now

  # check it
  batch_no = 1
  right = 0
  wrong = 0
  while batch_no > 0:
    predict = session.run(model_predict_predict, {model_predict_input: trainX})
    for i in range(batch_size):
      if np.argmax(predict[i]) == np.argmax(trainY[i]):
        right += 1
      else:
        wrong += 1
    X, Y, batch_no = next(batch)
    trainX, trainY = X, Y
    testX, testY = X, Y #overfit for now
  
  print("right({0}) wrong({1})".format(right, wrong))
'''
model = tflearn.DNN(net, tensorboard_verbose=0)
while 1: #training_iters
  model.fit(trainX, trainY, n_epoch=10, validation_set=(testX, testY), show_metric=True,
          batch_size=batch_size)
  _y=model.predict(X)
model.save("tflearn.lstm.model")
print (_y)
print (y)
'''

