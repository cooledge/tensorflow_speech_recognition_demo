'''
for the normal way the features are at the time slices. 
For this one I want to reverse it so the at each time all
the feature values are input

epoch 500
loaded batch of 2402 files
loaded batch of 2402 files
right(2322) wrong(110)

epochs = 20

version1: Using Adam            30% right
version2: Using GradientDescent 35% right


epochs = 50
Using GradientDescent
static_rnn
right(2216) wrong(632) Success rate is 40.0, dropout(90)
right(1964) wrong(884) Success rate is 20.0, dropout(70)
right(1827) wrong(1021) Success rate is 15.0, dropout(60)

epochs = 150

right(2520) wrong(328) Success rate is 30.0, dropout(90)

right(2328) wrong(520) Success rate is 35.0, dropout(70)

right(2250) wrong(598) Success rate is 25.0, dropout(60)


right(2338) wrong(510) Success rate is 30.0, dropout(80) n_mfcc_features(5)
right(2436) wrong(412) Success rate is 25.0, dropout(80) n_mfcc_features(10)
right(2463) wrong(385) Success rate is 15.0, dropout(80) n_mfcc_features(15)
right(2491) wrong(357) Success rate is 25.0, dropout(80) n_mfcc_features(20)
right(2455) wrong(393) Success rate is 30.0, dropout(80) n_mfcc_features(30)
right(2508) wrong(340) Success rate is 30.0, dropout(80) n_mfcc_features(40)

'''

from __future__ import division, print_function, absolute_import 
import speech_data
from audio_in import AudioIn
import tensorflow as tf
import numpy as np
import pdb
import os
import pickle
import argparse

parser = argparse.ArgumentParser(description='Speech recognition')
parser.add_argument('--ui', dest='ui', default=True)
parser.add_argument('--no-ui', dest='ui', action='store_false')
parser.add_argument('--test', default=False, action='store_true')
parser.add_argument('--epochs', type=int, default=0)
parser.add_argument('--mfcc_features', type=int, default=20)
parser.add_argument('--dropout', type=int, default=80)
args = parser.parse_args()

version_bidi = False

learning_rate = 0.0001
batch_size = 32

width = args.mfcc_features  # mfcc features
height = 120  # (max) length of utterance
classes = 10  # digits

# Network building
# (?, 20, 120)
model_input = tf.placeholder(tf.float32, shape=(None, width, height))
model_output = tf.placeholder(tf.float32, shape=(None, 10))

# (?, 128)
if version_bidi:
  cell = tf.nn.rnn_cell.BasicLSTMCell(128)
  cell_bw = tf.nn.rnn_cell.BasicLSTMCell(128)
else:
  cell = tf.nn.rnn_cell.BasicLSTMCell(128)
#cell = tf.contrib.rnn.TimeFreqLSTMCell(128)
model_dropout_prob = tf.placeholder_with_default(1.0, shape=())
cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=model_dropout_prob, output_keep_prob=model_dropout_prob)
rnn_input = [tf.squeeze(input, axis=2) for input in tf.split(model_input, height, axis=2)]

# rnn_input.shape = list(20) of 64X120
if version_bidi:
  rnn_output, rnn_state, rnn_state_bw = tf.nn.static_bidirectional_rnn(cell, cell_bw, rnn_input, dtype=tf.float32)
else:
  rnn_output, rnn_state = tf.nn.static_rnn(cell, rnn_input, dtype=tf.float32)
# rnn_output.shape =  list of 120 of 64X128
rnn_output = tf.concat(rnn_output, 1)

if version_bidi:
  model_fc_w = tf.get_variable("fc_w", shape=(2*height*128, 10))
else:
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

batch = speech_data.mfcc_batch_generator(batch_size, n_mfcc = width)
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
epochs = args.epochs
while epoch < epochs:
  epoch += 1
  print("epoch {0}".format(epoch))
  num_batches = int(len(trainX) / batch_size)
  batch_no = 1  # set to get in the loop
  while batch_no > 0:
    #print("batch_no({0})".format(batch_no))
    dropout_prob = float(args.dropout) / 100.0
    loss, _ = session.run([model_loss, model_train], {model_input: trainX, model_output: trainY, model_dropout_prob: dropout_prob})
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

def training_path():
  #return './data/training_wav'
  return './data/training'
  #return './data/training_pcm'

def get_test_files():
  path = training_path()
  return [os.path.join(path, name) for name in os.listdir(path)]

def do_test():
  test_files = get_test_files()
  right = 0
  wrong = 0
  for audio_filename in test_files:
    digit = int(os.path.basename(audio_filename).split('_')[0])
    audio_data = speech_data.mfcc_load_file(audio_filename, n_mfcc = width)
    predict = session.run(model_predict, {model_input: [audio_data]})
    details = [round(p*100) for p in predict[0]]
    predict = np.argmax(predict[0])
    if predict == digit:
      right += 1
    else: 
      wrong += 1
  return right/(right+wrong)*100

if args.test:
  print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX Success rate is {0}, dropout({1}) n_mfcc_features({2})".format(do_test(), args.dropout, args.mfcc_features))

class AI(AudioIn):

  def on_test(self):
    test_files = get_test_files()
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

if args.ui == 'on':
  ai = AI()
  ai.run() 

