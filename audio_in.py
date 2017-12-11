from sys import byteorder
from array import array
from struct import pack
from struct import unpack

import Tkinter as tk
import pyaudio
import wave
import os
import pdb

THRESHOLD = 500
if True:
  CHUNK_SIZE = 1024 # 1024 byte per array
  FORMAT = pyaudio.paInt16 
  #RATE = 44100
  RATE = 22050
else:
  CHUNK_SIZE = 512 # array size is 256 bytes
  FORMAT = pyaudio.paInt8 
  RATE = 8000

def load_wave(path):
    wf = wave.open(path, 'rb')
    data = wf.readframes(100000)
    unpacked_wave = unpack("<{0}h".format(len(data)/2), data)
    wf.close()
    return unpacked_wave

def is_silent(snd_data):
    "Returns 'True' if below the 'silent' threshold"
    #pdb.set_trace()
    #awave = load_wave('/home/dev/code/tensorflow_speech_recognition_demo/data/spoken_numbers_pcm/8_Princess_280.wav')
    #bwave = load_wave('./demo.wav')
    #print(snd_data)

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

def record():
    """
    Record a word or words from the microphone and 
    return the data as an array of signed shorts.

    Normalizes the audio, trims silence from the 
    start and end, and pads with 0.5 seconds of 
    blank sound to make sure VLC et al can play 
    it without getting chopped off.
    """
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=1, rate=RATE,
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

        silent = is_silent(snd_data)

        # pdb.set_trace()
        #break
        if silent and snd_started:
            num_silent += 1
        elif not silent and not snd_started:
            snd_started = True

        if snd_started and num_silent > 30:
            break

    sample_width = p.get_sample_size(FORMAT)
    stream.stop_stream()
    stream.close()
    p.terminate()

    r = normalize(r)
    r = trim(r)
    #r = add_silence(r, 0.5)
    return sample_width, r

def record_to_file(path):
    "Records from the microphone and outputs the resulting data to 'path'"
    sample_width, data = record()
    data = pack('<' + ('h'*len(data)), *data)


    wf = wave.open(path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(sample_width)
    wf.setframerate(RATE)
    wf.writeframes(data)
    wf.close()

'''
Ui with an activate button. When pressed audio is recorded. 
Once complete that is passed to a callback. The callback
returns a string which is displayed in the UI
'''

class AudioIn:

  def on_action_pressed(self):
    # get the audio file
    # do the callback
    print("please speak a word into the microphone")
    record_to_file('demo.wav')
    print("done - result written to demo.wav")
    self.set_text(self.on_audio('demo.wav'))

  def set_text(self, text):
    self.text.delete(1.0, tk.END);
    self.text.insert(tk.END, text)
   
  def training_path(self):
    return './data/training' 

  def get_start_instance(self):
    files = os.listdir(self.training_path())
    if len(files) == 0:
      return 0
    return max([int(file.split('_')[2].split('.')[0]) for file in files])

  def on_make_training_data_pressed(self):
    instance = self.get_start_instance()
    for digit in range(10):
      self.set_text("Say the number {0}".format(digit))
      record_to_file("{0}/greg_{1}_{2}.wav".format(self.training_path(), digit, instance))

  def on_audio(self, audio):
    return "the return text in the parent"

  def run(self):
    top = tk.Tk()

    action_button = tk.Button(top, text="Action", command=(lambda: self.on_action_pressed()))
    action_button.pack()

    action_button = tk.Button(top, text="Make Training Data", command=(lambda: self.on_make_training_data_pressed()))
    action_button.pack()

    self.text = tk.Text(top)
    self.text.insert(tk.INSERT, "This is where the text will appear")
    self.text.pack()

    top.mainloop()

'''
if __name__ == '__main__':
    print("please speak a word into the microphone")
    record_to_file('demo.wav')
    print("done - result written to demo.wav")
'''
   
if __name__ == '__main__':
  class AI(AudioIn):
    def on_audio(self, audio):
      return "on audio from the child"

  ai = AI()
  ai.run()

