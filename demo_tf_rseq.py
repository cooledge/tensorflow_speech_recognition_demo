'''
For this one train the nn on single digits but build 
a model that can recognize sequence of two digits

epoch 499
loaded batch of 2402 files
loaded batch of 2402 files
right(2359) wrong(73) both_right(2296) original_right(63)

next thing to try multiple rnns offset
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
classes = 10  # digits

# Network building
# (?, 20, 80)
model_input = tf.placeholder(tf.float32, shape=(batch_size, width, height))
model_output = tf.placeholder(tf.float32, shape=(batch_size, 10))
# net = tflearn.input_data([None, width, height])

# (?, 128)
cell = tf.nn.rnn_cell.BasicLSTMCell(128)
cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=0.8, output_keep_prob=0.8)
rnn_input = [tf.squeeze(input, axis=2) for input in tf.split(model_input, height, axis=2)]

# rnn_input.shape = list(80) of 64X20
rnn_output, rnn_state = tf.nn.static_rnn(cell, rnn_input, dtype=tf.float32)
# rnn_output.shape =  list of 80 of 64X128
rnn_output = tf.concat(rnn_output, 1)
# net = tflearn.lstm(net, 128, dropout=0.8)

model_fc_w = tf.get_variable("fc_w", shape=(height*128, 10))
model_fc_b = tf.get_variable("fc_b", shape=(10))
model_logits = tf.matmul(rnn_output, model_fc_w) + model_fc_b
# (? 10)
#net = tflearn.fully_connected(net, classes, activation='softmax')

smodel_input = tf.placeholder(tf.float32, shape=(batch_size, width, 2*height))
srnn_input = [tf.squeeze(input, axis=2) for input in tf.split(smodel_input, 2*height, axis=2)]
# srnn_output 160X64X128
srnn_output, srnn_state = tf.nn.static_rnn(cell, srnn_input, dtype=tf.float32)
# srnn_output 64*20480
srnn_output = tf.concat(srnn_output, 1)
smodel_logits1 = tf.matmul(tf.split(srnn_output, 2, axis=1)[0], model_fc_w) + model_fc_b
smodel_predict1 = tf.nn.softmax(smodel_logits1)
smodel_logits2 = tf.matmul(tf.split(srnn_output, 2, axis=1)[1], model_fc_w) + model_fc_b
smodel_predict2 = tf.nn.softmax(smodel_logits2)

model_loss = tf.losses.softmax_cross_entropy(model_output, model_logits)
opt = tf.train.AdamOptimizer(learning_rate)
model_train = opt.minimize(model_loss)
#net = tflearn.regression(net, optimizer='adam', learning_rate=learning_rate, loss='categorical_crossentropy')
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
  original_right = 0
  both_right = 0
  wrong = 0
  while batch_no > 0:
    train_X2 = np.zeros((64,20,160),dtype=float)
    train_X2[:,:,0:80] = trainX
    train_X2[:,:,80:160] = trainX
    predict1, predict2 = session.run([smodel_predict1, smodel_predict2], {smodel_input: train_X2})
    for i in range(batch_size):
      if np.argmax(predict1[i]) == np.argmax(trainY[i]) and np.argmax(predict2[i]) == np.argmax(trainY[i]):
        both_right += 1
      elif np.argmax(predict1[i]) == np.argmax(trainY[i]):
        original_right += 1
      else:
        wrong += 1
    X, Y, batch_no = next(batch)
    trainX, trainY = X, Y
    testX, testY = X, Y #overfit for now
  
  print("right({0}) wrong({1}) both_right({2}) original_right({3})".format(both_right+original_right, wrong, both_right, original_right))
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

