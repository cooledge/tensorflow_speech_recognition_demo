from sys import byteorder
from array import array
from struct import pack
from struct import unpack

import pyaudio
import wave
import os
import numpy as np
import pdb
import argparse
import aiy.assistant.grpc
import aiy.audio
import aiy.voicehat
import time

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

def record(nchannels=1):
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
    r = trim(r)
    #r = add_silence(r, 0.5)

    # split the channels
    ''' 
    pdb.set_trace()
    sample_len = int(len(r)/2)
    r = r[0:sample_len*2]
    r = np.reshape(r, (sample_len, 2))
    x = r[:, 0] - r[:, 1]
    r = r[:, 1]
    '''
    
    return sample_width, r

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

if __name__ == '__main__':
  pdb.set_trace()
  # wait for button press
  button = aiy.voicehat.get_button()
  button.wait_for_press()
  # sleep 3 seconds
  time.sleep(3) 
  # turn on light
  led = aiy.voicehat.get_led()
  led.set_state(aiy.voicehat.LED.ON)
  # record 2 channel
  path = './output.wav'
  record_to_file(path, 2)
  # turn off light
  led.set_state(aiy.voicehat.LED.OFF)
  # lag it to get the off to be sent
  time.sleep(1) 
