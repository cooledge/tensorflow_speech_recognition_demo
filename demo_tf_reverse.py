'''
for the normal way the features are at the time slices. 
For this one I want to reverse it so the at each time all
the feature values are input

epoch 500
loaded batch of 2402 files
loaded batch of 2402 files
right(2322) wrong(110)

Using Adam            30% right
Using GradientDescent 35% right

'''

from __future__ import division, print_function, absolute_import 
import speech_data
from audio_in import AudioIn
import tensorflow as tf
import numpy as np
import pdb
import os
import pickle

learning_rate = 0.0001
batch_size = 32

width = 20  # mfcc features
height = 80  # (max) length of utterance
classes = 10  # digits

# Network building
# (?, 20, 80)
model_input = tf.placeholder(tf.float32, shape=(None, width, height))
model_output = tf.placeholder(tf.float32, shape=(None, 10))

# (?, 128)
cell = tf.nn.rnn_cell.BasicLSTMCell(128)
model_dropout_prob = tf.placeholder_with_default(1.0, shape=())
cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=model_dropout_prob, output_keep_prob=model_dropout_prob)
rnn_input = [tf.squeeze(input, axis=2) for input in tf.split(model_input, height, axis=2)]

# rnn_input.shape = list(20) of 64X80
rnn_output, rnn_state = tf.nn.static_rnn(cell, rnn_input, dtype=tf.float32)
# rnn_output.shape =  list of 80 of 64X128
rnn_output = tf.concat(rnn_output, 1)

model_fc_w = tf.get_variable("fc_w", shape=(height*128, 10))
model_fc_b = tf.get_variable("fc_b", shape=(10))
model_logits = tf.matmul(rnn_output, model_fc_w) + model_fc_b

model_predict = tf.nn.softmax(model_logits)
model_l2_loss = tf.nn.l2_loss(model_fc_w) + tf.nn.l2_loss(model_fc_b)
model_loss = tf.losses.softmax_cross_entropy(model_output, model_logits) + model_l2_loss
#opt = tf.train.AdamOptimizer(learning_rate)
opt = tf.train.GradientDescentOptimizer(learning_rate)
model_train = opt.minimize(model_loss)

# Training

batch = speech_data.mfcc_batch_generator(batch_size)
X, Y, batch_no = next(batch)
trainX, trainY = X, Y

class Persistance:

  MODEL_NAME = 'tf_reverse'

  def __init__(self):
    self.saver = tf.train.Saver()
    self.checkpoint_path = "./saves/{0}.ckpt".format(self.MODEL_NAME)
    self.input_graph_path = "./saves/{0}.pbtxt".format(self.MODEL_NAME)
    self.pickle_file = "./saves/pickle_file"
    
  def load_graph(self, session):
    self.start = 0
    if os.path.exists(self.pickle_file):
      with open(self.pickle_file, 'rb') as input:
        props = pickle.load(input)
        self.start = props[ "epoch" ] + 1
      self.saver.restore(session, self.checkpoint_path)

  def save_graph(self,session, number_of_epochs=1):
    save_path = self.saver.save(session, self.checkpoint_path)

    with open(self.pickle_file, 'wb') as output:
      pickle.dump({ "epoch" : number_of_epochs+self.start }, output)
    print("Saved to {0}".format(save_path))

persistance = Persistance()

session = tf.Session()
session.run(tf.global_variables_initializer())

persistance.load_graph(session);

epoch = 0
epochs = 0
while epoch < epochs:
  epoch += 1
  print("epoch {0}".format(epoch))
  num_batches = int(len(trainX) / batch_size)
  batch_no = 1  # set to get in the loop
  while batch_no > 0:
    #print("batch_no({0})".format(batch_no))
    loss, _ = session.run([model_loss, model_train], {model_input: trainX, model_output: trainY, model_dropout_prob: 0.8})
    X, Y, batch_no = next(batch)
    trainX, trainY = X, Y

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
 
  persistance.save_graph(session) 
  print("right({0}) wrong({1})".format(right, wrong))

class AI(AudioIn):

  def on_test(self, test_files):
    right = 0
    wrong = 0
    message = ""
    for audio_filename in test_files:
      digit = int(os.path.basename(audio_filename).split('_')[0])
      audio_data = speech_data.mfcc_load_file(audio_filename)
      predict = session.run(model_predict, {model_input: [audio_data]})
      details = [round(p*100) for p in predict[0]]
      predict2 = session.run(model_predict, {model_input: [audio_data]})
      predict = np.argmax(predict[0])
      predict2 = np.argmax(predict2[0])
      if predict != predict2:
        pdb.set_trace()
      message += "predict({0}) digit({1}) - {2}\n".format(predict, digit, details)
      if predict == digit:
        right += 1
      else: 
        wrong += 1
    message += "{0} percent right".format(right/(right+wrong)*100)
    return message
      

  def on_audio(self, audio_filename):
    print("loading audio")
    audio_data = speech_data.mfcc_load_file(audio_filename)
    predict = session.run(model_predict, {model_input: [audio_data]})
    return "The prediction is {0}\nFull vector {1}".format(np.argmax(predict[0]), [round(p*100) for p in predict[0]])  

ai = AI()
ai.run() 

