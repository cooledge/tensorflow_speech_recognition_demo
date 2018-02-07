from sys import byteorder
from array import array
from struct import pack
from struct import unpack

import numpy
import pyaudio
import wave
import os
import numpy as np
import pdb
import argparse
import time
import wave
import math
import re

import pickle
import socket
import threading
import json
import requests

server_root_url  = 'http://dev-X555QA:5002/'

THRESHOLD = 500
if True:
  CHUNK_SIZE = 1024 # 1024 byte per array
  FORMAT = pyaudio.paInt16 
  #RATE = 44100
  RATE = 22050 # samples / second
else:
  CHUNK_SIZE = 512 # array size is 256 bytes
  FORMAT = pyaudio.paInt8 
  RATE = 8000

def load_wave(path):
  wf = wave.open(path, 'rb')
 # data = wf.readframes(wf.getnframes())
  data = wf.readframes(-1)
  unpacked_wave = unpack("<{0}h".format(len(data)/2), data)
  wf.close()
  return unpacked_wave

def plot_wave(path):
  import matplotlib
  import matplotlib.pyplot as plt

  '''
  path = './lr/right_coveeed.wav'
  path = './lr/right_wall.wav'
  path = './lr/right_open.wav'
  path = './lr/right_only_one.wav'
  '''

  wf = wave.open(path, 'rb')
  
  #Extract Raw Audio from Wav File
  n_to_read = wf.getnframes() - wf.getnframes() % wf.getnchannels()
  signal = wf.readframes(n_to_read)
  signal = np.fromstring(signal, 'Int16')

  signal = [ abs(s) for s in signal ]
  #Split the data into channels 
  channels = [[] for channel in range(wf.getnchannels())]
  for index, datum in enumerate(signal):
      channels[index%len(channels)].append(datum)

  #Get time from indices
  fs = wf.getframerate()
  Time=np.linspace(0, len(signal)/len(channels)/fs, num=len(signal)/len(channels))

  #diff = np.array(channels[0]) - np.array(channels[1])
  #channels = [diff]
  #Time=np.linspace(0, len(diff)/fs, num=len(diff))

  #Plot
  plt.figure(1)
  plt.title(path)
  for channel in channels:
      plt.plot(Time,channel)
  plt.show()
 
'''
# returns the left - right channel
def load_wave_diff(path):
  data = load_wave(path)
  sample_len = int(len(data)/2)
  r = r[0:sample_len*2]
  r = np.reshape(r, (sample_len, 2))
  x = r[:, 0] - r[:, 1]
  r = r[:, 1]
'''

def ms_to_n_samples(ms):
  # samples / (second * (1000 ms / second)) == samples per ms
  return int(RATE / 1000.0 * ms)
  
def is_silent(snd_data):
    "Returns 'True' if below the 'silent' threshold"
    return max(snd_data) < THRESHOLD

def normalize(snd_data):
    "Average the volume out"
    MAXIMUM = 16384
    times = float(MAXIMUM)/max(abs(i) for i in snd_data)

    r = array('h')
    for i in snd_data:
        r.append(int(i*times))
    return r

def trim(snd_data):
    "Trim the blank spots at the start and end"
    def _trim(snd_data):
        snd_started = False
        r = array('h')

        for i in snd_data:
            if not snd_started and abs(i)>THRESHOLD:
                snd_started = True
                r.append(i)

            elif snd_started:
                r.append(i)
        return r

    # Trim to the left
    snd_data = _trim(snd_data)

    # Trim to the right
    snd_data.reverse()
    snd_data = _trim(snd_data)
    snd_data.reverse()
    return snd_data

def add_silence(snd_data, seconds):
    "Add silence to the start and end of 'snd_data' of length 'seconds' (float)"
    r = array('h', [0 for i in xrange(int(seconds*RATE))])
    r.extend(snd_data)
    r.extend([0 for i in xrange(int(seconds*RATE))])
    return r

def record_ms(ms, n_channels=2):
  return record(n_channels, ms_to_n_samples(ms))[1]

def record(nchannels=1, max_samples=None):
    """
    Record a word or words from the microphone and 
    return the data as an array of signed shorts.

    Normalizes the audio, trims silence from the 
    start and end, and pads with 0.5 seconds of 
    blank sound to make sure VLC et al can play 
    it without getting chopped off.
    """
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=nchannels, rate=RATE,
        input=True, output=True,
        frames_per_buffer=CHUNK_SIZE)

    num_silent = 0
    snd_started = False

    r = array('h')

    while 1:
        # little endian, signed short
        snd_data = array('h', stream.read(CHUNK_SIZE))
        if byteorder == 'big':
            snd_data.byteswap()
        r.extend(snd_data)

        if max_samples is not None:
          if len(r) > max_samples:
            break
        else:
          print("got chunk")
          silent = is_silent(snd_data)

          if silent and snd_started:
              num_silent += 1
          elif not silent and not snd_started:
              snd_started = True

          if snd_started and num_silent > 30:
              break

    print("done")
    sample_width = p.get_sample_size(FORMAT)
    stream.stop_stream()
    stream.close()
    p.terminate()

    r = normalize(r)
    if max_samples is None:
      r = trim(r)
    #r = add_silence(r, 0.5)

    # split the channels
    ''' 
    sample_len = int(len(r)/2)
    r = r[0:sample_len*2]
    r = np.reshape(r, (sample_len, 2))
    x = r[:, 0] - r[:, 1]
    r = r[:, 1]
    '''
    
    return sample_width, r

'''
def record_to_file(path, nchannels=1):
    "Records from the microphone and outputs the resulting data to 'path'"
    sample_width, data = record(nchannels)
    data = pack('<' + ('h'*len(data)), *data)

    wf = wave.open(path, 'wb')
    wf.setnchannels(nchannels)
    wf.setsampwidth(sample_width)
    wf.setframerate(RATE)
    wf.writeframes(data)
    wf.close()
'''

def record_to_file(path, nchannels=1):
    "Records from the microphone and outputs the resulting data to 'path'"
    sample_width, data = record(nchannels)
    wave_to_file(path, data, nchannels, sample_width)

def add_index_to_filename(filename):
  i = 0
  while os.path.exists(filename + str(i) + ".wav"):
    i += 1
  return filename + str(i) + ".wav"

def wave_to_file(path, data, nchannels, sample_width):
    data = pack('<' + ('h'*len(data)), *data)

    wf = wave.open(path, 'wb')
    wf.setnchannels(nchannels)
    wf.setsampwidth(sample_width)
    wf.setframerate(RATE)
    wf.writeframes(data)
    wf.close()

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def left_path():
  return './lr/left.wav'

def rigth_path():
  return './lr/right.wav'

nn_batch_size = 20
slice_width_ms = 30
slice_stride_ms = 15
n_samples = ms_to_n_samples(slice_width_ms)*2
n_stride =  ms_to_n_samples(slice_stride_ms)*2
lstm_size = 128
number_of_classes = 3
cell_width = 1

# 100 epochs: right(650)/wrong(50) percent 0.928571428571

n_train_size_percent = 0.80

# slice up the input data into overlapping sections
# of length slice_ms using the data from file
#
# returns array of slices

def make_input(data):
  return [data[i:i+n_samples] for i in xrange(0, len(data)-n_samples, n_stride)]

def make_data():
  ensure_dir('./lr/train')
  ensure_dir('./lr/test')

  files = os.listdir('./lr')

  nn_input = []
  nn_output = []

  for file in files:
    if re.search('^left.*wav$', file) is None:
      output = [1,0,0]
    elif re.search('^right.*wav$', file) is None:
      output = [0,1,0]
    elif re.search('^silence.*wav$', file) is None:
      output = [0,0,1]
    else:
      continue
      
    data = load_wave('./lr/' + file)
    nn_input += make_input(data)
    nn_output += [output for i in range(len(nn_input))]

  # re-order the i/o
  indexes = [i for i in range(len(nn_input))]
  numpy.random.shuffle(indexes)
  nn_input = [nn_input[i] for i in indexes]
  nn_output = [nn_output[i] for i in indexes]

  n_train_size = int(len(nn_input) * n_train_size_percent)

  nn_train_input = nn_input[:n_train_size]
  nn_train_output = nn_output[:n_train_size]
 
  nn_test_input = nn_input[n_train_size:]
  nn_test_output = nn_output[n_train_size:]

  return [nn_train_input, nn_train_output, nn_test_input, nn_test_output]
 
def make_nn():
  #model_inputs = tf.placeholder(tf.float32, (nn_batch_size, n_samples), name='inputs')
  model_inputs = tf.placeholder(tf.float32, (None, n_samples), name='inputs')
  # class (left, right, neither)
  model_outputs = tf.placeholder(tf.float32, (nn_batch_size, number_of_classes), name='outputs')

  model_lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)

  model_rnn_inputs = tf.split(model_inputs, n_samples, axis=1)
  model_rnn_output, model_rnn_state = tf.nn.static_rnn(model_lstm, model_rnn_inputs, dtype=tf.float32)
  model_rnn_output = tf.concat(model_rnn_output, 1)

  model_w = tf.get_variable("fc_w", shape=(model_rnn_output.shape[1], number_of_classes))
  model_b = tf.get_variable("fc_b", shape=number_of_classes)
  model_logits = tf.matmul(model_rnn_output, model_w) + model_b

  model_predict = tf.nn.softmax(model_logits)
  model_loss = tf.losses.softmax_cross_entropy(model_outputs, model_logits)
  model_opt = tf.train.AdamOptimizer(0.001)
  model_train = model_opt.minimize(model_loss)

  return [model_inputs, model_outputs, model_train, model_loss, model_predict]
  
if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Record audio from microphones for training left right nn')
  parser.add_argument('--to_server', dest='to_server', type=str, default="", help='Send data to server side to a file with the given name')
  parser.add_argument('--left', dest='left', action='store_true', default=False, help='Sound will be coming from the left side')
  parser.add_argument('--right', dest='right', action='store_true', default=False, help='Sound will be coming from the right side')
  parser.add_argument('--make-data', dest='make_data', action='store_true', default=False, help='Convert the left right files into the training and test data')
  parser.add_argument('--listens', dest='listens', action='store_true', default=False, help='Server side listen and run through the neural net')
  parser.add_argument('--listenc', dest='listenc', action='store_true', default=False, help='Client side listen and run through the neural net')
  parser.add_argument('--get_data', dest='get_data', action='store_true', default=False, help='Run the server side of getting sample data')
  parser.add_argument('--plot', dest='plot', action='store_true', default=False, help='Plot the wavs in the lr directory.')
  parser.add_argument('--epochs', dest='epochs', type=int, default=True, help='Number of epochs to train')
  args = parser.parse_args()

  epochs = args.epochs
  # get the training data from the AIY speech device

  if args.plot:
    files = os.listdir('./lr')
    for file in files:
      plot_wave('./lr/' + file) 
   
  elif args.left or args.right or arg.to_server:
    import aiy.assistant.grpc
    import aiy.audio
    import aiy.voicehat
    # wait for button press
    ensure_dir('./lr')
    button = aiy.voicehat.get_button()
    button.wait_for_press()
    # sleep 3 seconds
    time.sleep(3) 
    # turn on light
    led = aiy.voicehat.get_led()
    led.set_state(aiy.voicehat.LED.ON)
    if args.to_server:
      max_samples = ms_to_n_samples(ms)
      sample_width, r = record(2, max_samples)

      json = {
        'name': args.to_server,
        'sample_width': sample_width,
        'nchannels': nchannels,
        'wav': sample
      }

      url = server_root_url + 'nn'
      response = requests.post(url, json) 
      print("Sent the data {0}".format(response))
    else:
      # record 2 channel
      if args.left:
        path = left_path()
      else:
        path = right_path()
      record_to_file(path, 2)
    # turn off light
    led.set_state(aiy.voicehat.LED.OFF)
    # lag it to get the off to be sent
    time.sleep(1) 
  elif args.make_data:
    print("Converting the left.wav and right.wav files into training data")
    #load_wave(left_path()) 
    #plot_wave(left_path())
    nn_train_input, nn_train_output, nn_test_input, nn_test_output = make_data()
    model_inputs, model_outputs, model_train, model_loss, model_predict = make_nn()

    session = tf.Session()
    session.run(tf.global_variables_initializer())

    save_path = "./left_right"
    save_name = 'cp'
    if not os.path.exists(save_path):
      os.makedirs(save_path)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
    if tf.train.latest_checkpoint(save_path):
      saver.restore(session, os.path.join(save_path, save_name))

    for epoch in range(epochs):
      print("Epoch {0}".format(epoch))
      average_loss = 0 
      n_batches = 0
      for batch_no in xrange(0, len(nn_train_input)-nn_batch_size, nn_batch_size):
        train_x = nn_train_input[batch_no:batch_no+nn_batch_size]
        train_y = nn_train_output[batch_no:batch_no+nn_batch_size]
        _, loss = session.run([model_train, model_loss], { model_inputs: train_x, model_outputs: train_y })
        average_loss += loss
        n_batches += 1

      print("Loss is {0}".format(average_loss/n_batches))
      if loss < 0.03:
        break

    saver.save(session, os.path.join(save_path, save_name))

    print("Doing the tests")
    right = 0
    wrong = 0
    for batch_no in xrange(0, len(nn_test_input)-nn_batch_size, nn_batch_size):
      test_x = nn_test_input[batch_no:batch_no+nn_batch_size]
      test_y = nn_test_output[batch_no:batch_no+nn_batch_size]
      predict = session.run(model_predict, { model_inputs: test_x })
      for i in range(len(predict)):
        if numpy.argmax(predict[i]) == numpy.argmax(test_y[i]):
          right += 1
        else:
          wrong += 1
      
    print("right({0})/wrong({1}) percent {2}".format(right, wrong, 1.0*right/(right+wrong)))
  elif args.get_data:
    app = Flask(__name__)
    api = Api(app)

    class Data(Resource):
      def post(self):
        name = request.json['name']
        sample_width = request.json['sample_width']
        nchannels = request.json['nchannels']
        wav = request.json['wav']
        path = add_index_to_filename("./lr/" + filename)
        save_to_file(path, wav, nchannels, sample_width)
        plot_wave(path)

    api.add_resource(Data, '/data')
    app.run(port='5002', host='0.0.0.0')
  elif args.listens:
   
    import tensorflow as tf
    from flask import Flask, request
    from flask_restful import Resource, Api
    from flask_jsonpify import jsonify

    print('Loading data') 
    nn_train_input, nn_train_output, nn_test_input, nn_test_output = make_data()
    print('Building model') 
    model_inputs, model_outputs, model_train, model_loss, model_predict = make_nn()
    print('Init complete') 
    
    session = tf.Session()
    session.run(tf.global_variables_initializer())

    save_path = "./left_right"
    save_name = 'cp'
    if not os.path.exists(save_path):
      os.makedirs(save_path)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
    if tf.train.latest_checkpoint(save_path):
      saver.restore(session, os.path.join(save_path, save_name))
  
    app = Flask(__name__)
    api = Api(app)

    class NN(Resource):
      def post(self):
        wav = request.json['wav']
        #pdb.set_trace()
        #checksum = sum(wav)
        #print("The sum is {0}".format(checksum))
        test_x = [wav]
        predict = session.run(model_predict, { model_inputs: test_x })
        p = numpy.argmax(predict[0])
        print(predict[0]) 
        return jsonify([p])

    api.add_resource(NN, '/nn')
    app.run(port='5002', host='0.0.0.0')
  elif args.listenc:
    while True:
      pdb.set_trace() 
      sample = record_ms(slice_width_ms)
      n_samples = ms_to_n_samples(slice_width_ms)*2
      sample = sample[:n_samples]
      sample = sample.tolist()
      url = server_root_url + 'nn'
      response = requests.post(url, json={'wav': sample}) 
      p = response.json()[0]
      
      if p == 0:
        print("Left")
      elif p == 1:
        print("Right")
      else:
        print("Silence")
