import os
import sys
import numpy as np
import glob
import json
import wave
import contextlib
import soundfile as sf
from shutil import copyfile
def gen_json():
  with open("vocab.json", encoding = "utf8") as v_file: 
    v = v_file.readline()
    vocab = json.loads(v)
  partitions = ["test", "train", "dev"]
  main_path = "./data_thchs30/"

  for part in partitions:
    path = os.path.abspath(os.path.join(main_path, part))
    os.chdir(path)
    files = glob.glob(os.path.abspath("*.wav"))
    js = []
    for f in files:
      d = {}
      audio_path = os.path.abspath(f)
      audio = sf.SoundFile(audio_path)
      duration = int(10**6 * len(audio) / audio.samplerate)
      trn = os.path.abspath(f+ ".trn")
      label = []
      with open(trn, encoding = "utf8") as file:
        label_path = file.readline()[0:-1]
        abs_path = os.path.abspath(label_path)
        with open(abs_path, encoding = "utf8") as label_file:
          l =label_file.readline()[0:-1]
          for w in l:
            if w == " ":
              continue
            else:
              label.append(vocab[w])
      d["audio"] = audio_path
      d["text"] = label
      d["duration"] = duration
      js.append(d)
      
    os.chdir(os.path.abspath("../../data/partitions"))
    with open(part + ".json", "w") as outfile:
      outfile.write(json.dumps(js))
    os.chdir(os.path.abspath("../../"))

def move_data():
  partitions = ["test", "train", "dev"]
  main_path = "./data_thchs30/"

  for part in partitions:
    path = os.path.abspath(os.path.join(main_path, part))
    os.chdir(path)
    files = glob.glob(os.path.abspath("*.wav"))
    direct = os.path.abspath(os.path.join("../../data/", part))
    for f in files:
      suf = f.split("/")[-1]
      print("moving", part, suf)
      copyfile(f, os.path.abspath(os.path.join(direct,suf)))
    os.chdir(os.path.abspath("../../"))
      

if __name__ == "__main__":
  gen_json()
  move_data()

