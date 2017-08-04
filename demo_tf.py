from __future__ import division, print_function, absolute_import
#import speech_data
import tensorflow as tf
import pdb

'''
# list of seq_len of array of size batch_size, state_size
seq_len = 3
batch_size = 5
state_size = 2

input = [
tf.constant([[1,1],[2,2],[3,3],[4,4],[5,5]]),
tf.constant([[1,1],[2,2],[3,3],[4,4],[5,5]]),
tf.constant([[1,1],[2,2],[3,3],[4,4],[5,5]])
]

pdb.set_trace()

session = tf.Session()
session.run(tf.concat(input, 1))
'''

learning_rate = 0.0001
training_iters = 300000  # steps
batch_size = 64

width = 20  # mfcc features
height = 80  # (max) length of utterance
classes = 10  # digits

'''
batch = word_batch = speech_data.mfcc_batch_generator(batch_size)
X, Y = next(batch)
trainX, trainY = X, Y
testX, testY = X, Y #overfit for now
'''



# Network building
# (?, 20, 80)
model_input = tf.placeholder(tf.float32, shape=(batch_size, width, height))
# net = tflearn.input_data([None, width, height])

# (?, 128)
cell = tf.nn.rnn_cell.BasicLSTMCell(128)
cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=0.8, output_keep_prob=0.8)
rnn_input = [tf.squeeze(input, axis=1) for input in tf.split(model_input, width, axis=1)]

pdb.set_trace()
rnn_output, rnn_state = tf.nn.static_rnn(cell, rnn_input, dtype=tf.float32)
rnn_output = tf.concat(rnn_output, 1)
# net = tflearn.lstm(net, 128, dropout=0.8)
fc_w = tf.get_variable("fc_w", shape=(128, 10))
fc_b = tf.get_variable("fc_b", shape=(10))
logits = tf.matmul(rnn_output, fc_w) + fc_b
# rrr = tf.concat([tf.constant([[1],[3]]), tf.constant([[2],[4]])],1)
# (? 10)
net = tflearn.fully_connected(net, classes, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=learning_rate, loss='categorical_crossentropy')
# Training

### add this "fix" for tensorflow version errors
col = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
for x in col:
    tf.add_to_collection(tf.GraphKeys.VARIABLES, x ) 


model = tflearn.DNN(net, tensorboard_verbose=0)
while 1: #training_iters
  model.fit(trainX, trainY, n_epoch=10, validation_set=(testX, testY), show_metric=True,
          batch_size=batch_size)
  _y=model.predict(X)
model.save("tflearn.lstm.model")
print (_y)
print (y)
