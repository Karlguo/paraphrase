from nltk.corpus import stopwords
from nltk.corpus import wordnet
from itertools import chain
import os
import sys
import random

length_limit = 20

fin = open(sys.argv[1], "r", encoding="utf-8")
fout = open(sys.argv[2], "w", encoding="utf-8")

stop_words = set(stopwords.words('english'))

count = 0
rcount = 0
wcount = 0
cwcount = 0
for line in fin.read().split("\n")[:-1]:
    count += 1
    s_line = line.split("\t")
    sp_line = s_line[0].split()
    #sp_line = line.split()
    if len(sp_line) < length_limit:
        rcount += 1
        new_line = [w for w in sp_line if w not in stop_words]
        syns = []
        for w in new_line:
            wcount += 1
            cwcount += 1
            lemmas = list(set(chain.from_iterable([word.lemma_names() for word in wordnet.synsets(w)])))
            lemmas = [w.lower() for w in lemmas if "-" not in w and "_" not in w]
            if len(lemmas) == 0:
                cwcount -= 1
                lemmas = [w]
            syns.append(lemmas)
        if len(syns) == 0: continue
        newnewline = [random.choice(ws) for ws in syns]
        #random.shuffle(newnewline)
        fout.write(" ".join(newnewline))
        fout.write("\t" + s_line[0])
        for j in range(1, len(s_line)):
            fout.write("\t" + s_line[j])
        fout.write("\n")

print (count, rcount)
print (wcount, cwcount)
