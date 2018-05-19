import os
import sys
import numpy as np

def proc(addr):
    os.chdir(os.path.abspath("./data_thchs30/" + addr))
    names = None
    with open(".wav.scp") as f:
        names = f.readlines()
    n = len(names)
    for i in range(n):
        name = names[i].split()[1]
        readname = name + ".trn"
        line = None 
        with open(readname, encoding="utf8") as g:
            line = g.readline()[0:-1]
        words_arr = []
        for w in line:
            if w == " ":
                continue
            else:
                words_arr.append(ord(w))
        writename = name + ".zh"
        np.savetxt(writename, words_arr, delimiter= ',')
    os.chdir(os.path.abspath("../../"))

if __name__ == "__main__":
    proc("data")