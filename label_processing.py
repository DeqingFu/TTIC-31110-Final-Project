import os
import sys
import numpy as np
import glob

MAX_LEN = 49
def proc(addr):
    os.chdir(os.path.abspath("./data_thchs30/" + addr))
    names = glob.glob(os.path.abspath("*.wav"))
    cnt = 0
    for name in names:
        readname = name + ".trn"
        line = None 
        with open(readname, encoding = "utf8") as g:
            line = g.readline()[0:-1]
        words_arr = []
        for w in line:
            if w == " ":
                continue
            else:
                words_arr.append(ord(w))
        words_arr.append(1) # end of sentence marker
        words_arr = np.asanyarray(words_arr)
        n = len(words_arr)
        pad = MAX_LEN - n
        words_arr = np.concatenate((words_arr, np.zeros(pad)))
        writename = name + ".zh"
        np.savetxt(writename, words_arr, delimiter= ',', fmt='%.8e')
        if cnt%100 == 0:
            print("processing", cnt)
        cnt += 1
    os.chdir(os.path.abspath("../../"))

if __name__ == "__main__":
    proc("data")