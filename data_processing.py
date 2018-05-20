import librosa
import librosa.display
import os, sys
import matplotlib.pyplot as plt
import numpy as np
import glob
N_MFCC = 13
T_MAX = 703

def get_pitch(p, mag, t):
  index = mag[:,t].argmax()
  pitch = p[index, t]
  return pitch

def proc(addr):
    os.chdir(os.path.abspath("./data_thchs30/" + addr))
    names = glob.glob(os.path.abspath("*.wav"))
    cnt = 0
    for filename in names:
      # extract data 
      y, sr = librosa.load(filename)

      # compute mfcc data, flatten to one 1d array
      ## get data
      mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc= N_MFCC)
      mfcc_delta = librosa.feature.delta(mfcc)
      mfcc_delta2 = librosa.feature.delta(mfcc_delta)
      ## padding
      r,c = np.shape(mfcc)
      pad = N_MFCC * T_MAX - r*c
      mfcc = np.append(np.ndarray.flatten(mfcc), np.zeros(pad))
      mfcc_delta = np.append(np.ndarray.flatten(mfcc_delta), np.zeros(pad))
      mfcc_delta2 = np.append(np.ndarray.flatten(mfcc_delta2), np.zeros(pad))
      mfcc_data = np.concatenate((mfcc, mfcc_delta, mfcc_delta2))
      
      # compute energy data, using root-mean-square energy for each fram
      ## get data
      energy = librosa.feature.rmse(y=y)
      energy_delta = librosa.feature.delta(energy)
      energy_delta2 = librosa.feature.delta(energy_delta)
      ## padding
      r,c = np.shape(energy) 
      pad = T_MAX - r*c 
      energy = np.append(energy, np.zeros(pad))
      energy_delta = np.append(energy_delta, np.zeros(pad))
      energy_delta2 = np.append(energy_delta2, np.zeros(pad))
      energy_data = np.concatenate((energy, energy_delta, energy_delta2))

      # compute pitch data
      ## get data
      p, mag = librosa.piptrack(y=y, sr=sr)
      _, T = np.shape(p)
      pitches = np.asarray(list(get_pitch(p, mag, t) for t in range(T)))
      pitches_delta = librosa.feature.delta(pitches)
      pitches_delta2 = librosa.feature.delta(pitches_delta)
      ## padding
      pad = T_MAX - T 
      pitches = np.append(pitches, np.zeros(pad))
      pitches_delta = np.append(pitches_delta, np.zeros(pad))
      pitches_delta2 = np.append(pitches_delta2, np.zeros(pad))
      pitches_data = np.concatenate((pitches, pitches_delta, pitches_delta2))

      # combine and save to text
      data = np.concatenate((mfcc_data, energy_data, pitches_data))
      writename = filename + ".data"
      np.savetxt(writename, data, delimiter=",",  fmt='%.8e')

      if cnt%20 == 0:
        print("processing", cnt)
      cnt+=1


if __name__ == "__main__":
  proc("data") 