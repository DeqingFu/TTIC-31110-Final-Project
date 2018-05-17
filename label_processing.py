import os
import sys

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
        writename = name + ".zh"
        with open(writename, "w", encoding = 'utf8') as h:
            h.write(line)
    os.chdir(os.path.abspath("../../"))

if __name__ == "__main__":
    proc("data")