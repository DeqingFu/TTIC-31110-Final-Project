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
    cnt = 0
    for filename in names:
      # extract data 
      y, sr = librosa.load(filename)

      # compute mfcc data, flatten to one 1d array
      ## get data
      mfcc = librosa.feature.mfcc(y=y, sr=sr)
      mfcc_delta = librosa.feature.delta(mfcc)
      mfcc_delta2 = librosa.feature.delta(mfcc_delta)
      ## padding
      r,c = np.shape(mfcc)
      pad = 10300 - r*c #10300 is the max len of product of shape of mfcc
      mfcc = np.append(np.ndarray.flatten(mfcc), np.zeros(pad))
      mfcc_delta = np.append(np.ndarray.flatten(mfcc_delta), np.zeros(pad))
      mfcc_delta2 = np.append(np.ndarray.flatten(mfcc_delta2), np.zeros(pad))
      mfcc_data = np.ndarray.flatten(np.asarray([mfcc, mfcc_delta, mfcc_delta2]))
      
      # compute energy data, using root-mean-square energy for each fram
      ## get data
      energy = librosa.feature.rmse(y=y)
      energy_delta = librosa.feature.delta(energy)
      energy_delta2 = librosa.feature.delta(energy_delta)
      ## padding
      r,c = np.shape(energy) 
      pad = 515 - r*c # 515 the max len for energy
      energy = np.append(energy, np.zeros(pad))
      energy_delta = np.append(energy_delta, np.zeros(pad))
      energy_delta2 = np.append(energy_delta2, np.zeros(pad))
      energy_data = np.ndarray.flatten(np.asarray([energy, energy_delta, energy_delta2]))
      
      # compute pitch data
      ## get data
      pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
      pitches_delta = librosa.feature.delta(pitches)
      pitches_delta2 = librosa.feature.delta(pitches_delta)
      ## padding
      r,c = np.shape(pitches)
      pad = 527875 - r * c #527875 is the max len for energy
      pitches = np.append(np.ndarray.flatten(pitches), np.zeros(pad))
      pitches_delta = np.append(np.ndarray.flatten(pitches_delta), np.zeros(pad))
      pitches_delta2 = np.append(np.ndarray.flatten(pitches_delta2), np.zeros(pad))
      pitches_data = np.ndarray.flatten(np.asarray([pitches, pitches_delta,pitches_delta2]))

      # combine and save to text
      data = np.concatenate((mfcc_data, energy_data, pitches_data))
      writename = filename + ".data"
      np.savetxt(writename, data, delimiter=",")
      if cnt%100 == 0:
        print(cnt)
      cnt+=1

if __name__ == "__main__":
  proc("data") 