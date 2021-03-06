import os
import sys
import numpy as np
import glob
import json
import wave
import contextlib
import soundfile as sf
from shutil import copyfile
from shutil import copy
def gen_json():
  with open("vocab.json", encoding = "utf8") as v_file: 
    v = v_file.readline()
    vocab = json.loads(v)
  partitions = ["test", "dev", "train"]
  main_path = "./data_thchs30/"
  jsons = []
  for part in partitions:
    path = os.path.abspath(os.path.join(main_path, part))
    os.chdir(path)
    files = glob.glob(os.path.abspath("*.wav"))
    js = []
    for f in files:
      d = {}
      suf = f.split("\\")[-1]
      audio_path = os.path.join(os.path.join(os.path.abspath("../../rnn/data"), part), suf)
      audio = sf.SoundFile(audio_path)
      duration = int(10**6 * len(audio) / audio.samplerate)
      trn = os.path.abspath(f+ ".trn")
      label = []
      with open(trn, encoding = "utf8") as file:
        label_path = file.readline()[0:-1]
        abs_path = os.path.abspath(label_path)
        with open(abs_path, encoding = "utf8") as label_file:
          _ = label_file.readline()
          label = label_file.readline()[0:-1].split()
      d["audio"] = audio_path
      d["text"] = label
      d["duration"] = duration
      js.append(d)
    if part != "train":
      jsons.append(js)
    else:
      js += jsons[0]
      js += jsons[1]
      
    os.chdir(os.path.abspath("../../rnn/data/partitions"))
    with open(part + ".json", "w") as outfile:
      for j in js:
        outfile.write(json.dumps(j)+'\n')
    os.chdir(os.path.abspath("../../../"))

def move_data():
  partitions = ["test", "dev", "train"]
  main_path = "./data_thchs30/"

  for part in partitions:
    
    path = os.path.abspath(os.path.join(main_path, part))
    os.chdir(path)
    files = glob.glob(os.path.abspath("*.wav"))
    direct = os.path.abspath(os.path.join("../../rnn/data/", part))
    for f in files:
      suf = f.split("\\")[-1]
      print("moving", part, suf)
      copyfile(f, os.path.abspath(os.path.join(direct,suf)))
    os.chdir(os.path.abspath("../../"))
    '''
    if part == "train":
      direct = os.path.abspath("./rnn/data/train")
      os.chdir(os.path.abspath("./rnn/data/test"))
      files = glob.glob(os.path.abspath("*.wav"))
      for f in files:
        suf = f.split("\\")[-1]
        print("moving", part, suf)
        copy(f, direct)
      
      os.chdir(os.path.abspath("../dev"))
      files = glob.glob(os.path.abspath("*.wav"))
      for f in files:
        suf = f.split("\\")[-1]
        print("moving", part, suf)
        copy(f, direct)
    os.chdir(os.path.abspath("../../../"))
    '''
      
      

      

if __name__ == "__main__":
  move_data()
  gen_json()

