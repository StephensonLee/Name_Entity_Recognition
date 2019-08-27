# filter the files with high percentage of numbers
from nltk import word_tokenize
import pandas as pd

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
def file_remove_num(PCT):
    filesout=[]
    for file in files:
        num = 0
        txt = open(path + '/' + file).read()
        for word in word_tokenize(txt):
            if is_number(word):
                num += 1
        if num / len(word_tokenize(txt)) < PCT/100:
            filesout.append(file)
            # print(file)
        else:
            print(file)
    return filesout

# get all the txt file names with low number percentage
path = "CCAT"
PCT = 10
files = pd.read_csv(open('Names.csv')).values
files = [x for [x] in files]
filesout = file_remove_num(PCT)

# save the organization names into results variable
pd.DataFrame(filesout).to_csv('Names_Numpct_'+str(PCT)+'.csv', index=False, header=None)

# read file and print the names
# filename=pd.read_csv(open('Numpct10.csv'),header=None).values
# filename = [a for [a] in filename]
# for i in filename:
#     print(i)
