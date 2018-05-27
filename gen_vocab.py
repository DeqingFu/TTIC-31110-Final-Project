import os
import sys
import numpy as np
import glob
import json
def proc(addr):
  os.chdir(os.path.abspath("./data_thchs30/" + addr))
  names = glob.glob(os.path.abspath("*.wav"))
  d = {}
  dict_len = 0
  for name in names:    
    print("processing", name.split('.')[0])
    readname = name + ".trn"
    line = None 
    with open(readname, encoding = "utf8") as g:
      line = g.readline()[0:-1]
    for w in line:
      if w == " ":
        continue
      elif w in d:
        continue
      else:
        dict_len += 1
        d[w] = dict_len
  os.chdir(os.path.abspath("../../"))
  return d, dict_len

if __name__ == "__main__":
  d, d_len = proc("data")
  with open("vocab.json", "w", encoding = "utf8") as file:
    file.write(json.dumps(d))