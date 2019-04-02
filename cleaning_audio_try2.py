# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 18:52:13 2019

@author: B.Sai Ashish
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 18:28:27 2019

@author: B.Sai Ashish
"""

import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from python_speech_features import mfcc, logfbank

def plot_signals(signals):
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=False,
                             sharey=True, figsize=(20,5))
    fig.suptitle('Time Series', size=10)
    i = 0
    for x in range(1):
        for y in range(0):
            axes[x,y].set_title(list(signals.keys())[i])
            axes[x,y].plot(list(signals.values())[i])
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1

def plot_fft(fft):
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=False,
                             sharey=True, figsize=(20,5))
    fig.suptitle('Fourier Transforms', size=16)
    i = 0
    for x in range(1):
        for y in range(0):
            data = list(fft.values())[i]
            Y, freq = data[0], data[1]
            axes[x,y].set_title(list(fft.keys())[i])
            axes[x,y].plot(freq, Y)
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1

def plot_fbank(fbank):
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=False,
                             sharey=True, figsize=(20,5))
    fig.suptitle('Filter Bank Coefficients', size=16)
    i = 0
    for x in range(1):
        for y in range(0):
            axes[x,y].set_title(list(fbank.keys())[i])
            axes[x,y].imshow(list(fbank.values())[i],
                    cmap='hot', interpolation='nearest')
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1

def plot_mfccs(mfccs):
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=False,
                             sharey=True, figsize=(20,5))
    fig.suptitle('Mel Frequency Cepstrum Coefficients', size=16)
    i = 0
    for x in range(1):
        for y in range(0):
            axes[x,y].set_title(list(mfccs.keys())[i])
            axes[x,y].imshow(list(mfccs.values())[i],
                    cmap='hot', interpolation='nearest')
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1
            
            
    

def envelope(y, rate, threshold):
  mask = []
  y = pd.Series(y).apply(np.abs)
  y_mean = y.rolling(window=int(rate/10), min_periods=1, center = True).mean()
  for mean in y_mean:
    if mean>threshold:
      mask.append(True)
    else:
      mask.append(False)
      
  return mask    

def calc_fft(y, rate):
  n = len(y)
  freq = np.fft.rfftfreq(n, d=1/rate)
  Y = abs(np.fft.rfft(y)/n)
  return(Y,freq)
  
df = pd.read_csv('instruments_try2.csv')

df.set_index('fname', inplace=True)

for f in df.index:
  rate, signal = wavfile.read('wavfiles_try2/'+f)
  df.at[f,'length'] = signal.shape[0]/rate

classes = list(np.unique(df.label))
class_dist = df.groupby(['label'])['length'].mean()

fig, ax = plt.subplots()
ax.set_title('Class Distribution', y=1.08)
ax.pie(class_dist, labels=class_dist.index, autopct='%1.1f%%',shadow=False, startangle=90)
ax.axis('equal')
plt.show()
df.reset_index(inplace=True)

signals = {}
fft = {}
fbank = {}
mfccs = {}

import librosa
for c in classes:
  print (c)  
  wav_file = df[df.label == c].iloc[0,0]
  signal, rate = librosa.load('wavfiles_try2/'+wav_file, sr=44100)
  mask = envelope(signal, rate, 0.0005)
  signal = signal[mask]
  signals[c] = signal
  fft[c] = calc_fft(signal, rate)
  
  bank = logfbank(signal[:rate], rate, nfilt=26, nfft=1103).T
  fbank[c] = bank
  mel = mfcc(signal[:rate], rate, numcep = 13, nfilt=26, nfft=1103).T
  mfccs[c] = mel
  
print('signals')  
plot_signals(signals)
plt.show()

print('fft')
plot_fft(fft)
plt.show()

print('fbank')
plot_fbank(fbank)
plt.show()

print('mfcss')
plot_mfccs(mfccs)
plt.show()

if len(os.listdir('clean_try2')) == 0:
  for f in tqdm(df.fname):
    signal, rate = librosa.load('wavfiles_try2/'+f, sr=16000)
    mask = envelope(signal, rate, 0.0005)
    wavfile.write(filename='clean_try2/'+f, rate=rate, data=signal[mask])


  
