import librosa
import librosa.display
import os, sys
import matplotlib.pyplot as plt
import numpy as np

def proc(addr):
    os.chdir(os.path.abspath("./data_thchs30/" + addr))
    names = None
    with open(".wav.scp") as f:
        names = f.readlines()
    names = list(map(lambda name: name.split()[1], names))
    for filename in names:
      # extract data 
      y, sr = librosa.load(filename)
      
      # compute mfcc data, flatten to one 1d array
      mfcc = librosa.feature.mfcc(y=y, sr=sr)
      mfcc_delta = librosa.feature.delta(mfcc)
      mfcc_delta2 = librosa.feature.delta(mfcc_delta)
      mfcc_data = np.ndarray.flatten(np.asarray([mfcc, mfcc_delta, mfcc_delta2]))
      
      # compute energy data, using root-mean-square energy for each fram
      energy = librosa.feature.rmse(y=y)
      energy_delta = librosa.feature.delta(energy)
      energy_delta2 = librosa.feature.delta(energy_delta)
      energy_data = np.ndarray.flatten(np.asarray([energy, energy_delta, energy_delta2]))

      # compute pitch data
      pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
      pitches_delta = librosa.feature.delta(pitches)
      pitches_delta2 = librosa.feature.delta(pitches_delta)
      pitches_data = np.ndarray.flatten(np.asarray([pitches, pitches_delta,pitches_delta2]))
      

if __name__ == "__main__":
  proc("data") 