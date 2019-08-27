import os
import pandas as pd
from nltk import word_tokenize
from simhash import Simhash


# the method to calculatet the similarity of the two text
def simhash_similarity(text1, text2):
    aa_simhash = Simhash(text1)
    bb_simhash = Simhash(text2)
    max_hashbit = max(len(bin(aa_simhash.value)), (len(bin(bb_simhash.value))))
    distince = aa_simhash.distance(bb_simhash)
    similar = 1 - distince / max_hashbit
    return similar


# remove similar articles
def remove_similar(files):
    filessm = files.copy()
    for i, name1 in enumerate(files):
        for j, name2 in enumerate(files):
            if j > i:
                txt1 = open(path + '/' + name1)
                txt2 = open(path + '/' + name2)
                similarity = simhash_similarity(txt1, txt2)
                if similarity > 0.9:
                    print(name1, name2)
                    if name2 in filessm:
                        filessm.remove(name2)
    return filessm


# judge wether the str is a number
def is_number(str):
    try:
        if str == 'NaN':
            return False
        float(str)
        return True
    except ValueError:
        return False


# remove the files with high numbers percentage
def remove_numfile(files, PCT):
    filesout = []
    for file in files:
        num = 0
        txt = open(path + '/' + file).read()
        for word in word_tokenize(txt):
            if is_number(word):
                num += 1
        if num / len(word_tokenize(txt)) < PCT / 100:
            filesout.append(file)
            # print(file)
        else:
            print(file)
    return filesout


# enumerate the similarity of text pairs
path = "CCAT"
PCT = 10
files = os.listdir(path)

# save the reults into filessm with removing the similar text
files = remove_similar(files)
# get all the txt file names with number percentage lower than PCT/100
filesout = remove_numfile(files, PCT)

# save the results into csv files for further analyse
pd.DataFrame(filesout).to_csv('Preprecessing.csv', header=False, index=None)
