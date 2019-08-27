from nltk import ne_chunk, pos_tag, word_tokenize, sent_tokenize, FreqDist
from nltk.tree import Tree
from datetime import datetime
import os
import pandas as pd
import numpy as np


# define the function to extracts name entities phrase
def get_continuous_chunks(text):
    # sent_count = len(sent_tokenize(text))
    # print(sent_count)
    chunked = ne_chunk(pos_tag(word_tokenize(text)))
    continuous_chunk = []
    # all_chunk variable save repeat words multi times
    all_chunk = []
    # continuous_chunk variable save repeat words once only
    current_chunk = []
    # count the organiztion number in the text
    num = 0
    # if sent_count > 4:
    # extact the organization and concatenate them into phrase
    for item in chunked:
        if type(item) == Tree and item.label() == 'ORGANIZATION':
            current_chunk.append(" ".join([token for token, pos in item.leaves()]))
        elif current_chunk:
            named_entity = " ".join(current_chunk)
            all_chunk.append(named_entity)
            num += 1
            if named_entity not in continuous_chunk:
                continuous_chunk.append(named_entity)
            current_chunk = []
        else:
            continue
    # take account of the possibility of last word being an Organization
    if current_chunk:
        named_entity = ' '.join(current_chunk)
        all_chunk.append(named_entity)
        if named_entity not in continuous_chunk:
            continuous_chunk.append(named_entity)
    return all_chunk, continuous_chunk, num


# get the files containing the organizations
def get_filenames(target):
    filenames = []
    current_chunk = []
    for file in files:
        fp = open(path + '/' + file)
        txt = fp.read()
        sent_tokens = sent_tokenize(txt)
        # if len(sent_tokens) > 4:
        for item in ne_chunk(pos_tag(word_tokenize(txt))):
            if type(item) == Tree and item.label() == 'ORGANIZATION':
                current_chunk.append(" ".join([token for token, pos in item.leaves()]))
            elif current_chunk:
                named_entity = " ".join(current_chunk)
                if target == named_entity:
                    filenames.append(file)
                current_chunk = []
            else:
                continue
        if current_chunk:
            named_entity = ' '.join(current_chunk)
            if target == named_entity:
                filenames.append(file)
    return filenames


# generate abbreviation of a organization
def abbr(str):
    abbr = ''
    for i in word_tokenize(str):
        abbr += i[0]
    return abbr


# remove the alias like govett,inc,re,corp,plc,ltd,corp,co
def remove_alias(map):
    list = sorted(map.keys(), key=lambda i: len(i), reverse=True)
    print(list)
    # print(map)
    print(len(map.keys()))
    banlist = ['bank', 'national', 'international', 'american']

    for i, org1 in enumerate(list):
        for j, org2 in enumerate(list):
            if j > i:
                if org1.istitle():
                    if not org2.isupper():
                        org1_token = word_tokenize(org1)
                        org2_token = word_tokenize(org2)
                        if org2.lower() not in banlist and org1[:len(org2)] == org2:
                            if org2_token[0] == org1_token[0]:
                                map[org2] += map[org1]
                                map[org1] = 0
                                print(org1, org2)
                                continue
                    elif len(org2) > 2 and org2 == abbr(org1):
                        map[org2] += map[org1]
                        map[org1] = 0
                        print(org1, org2)
                        continue
                if org1.lower() == org2.lower():
                    map[org2] += map[org1]
                    map[org1] = 0
                    print(org1, org2)
    sorted_map = sorted(map.items(), key=lambda i: i[1], reverse=True)
    return (sorted_map)


# get all the txt file names for iteration
path = "CCAT"
# files = os.listdir(path)

# read the files without identicle items
files = pd.read_csv('Names_Numpct_10.csv', header=None).values
files = [x for [x] in files]

# # read the files containing perticular organization
# filename = get_filenames('BET')
# pd.DataFrame(filename).to_csv('test.csv')

# save the organization names into results variable
results = []
# record the number of organizations in the article
count = []
# record the start time of the experiment
start = datetime.now()

# read the txt file one by one and retrieve the organizations
for file in files:
    print(path + '/' + file)
    fp = open(path + '/' + file)
    txt = fp.read()
    all_chunked, continuous_chunked, curcount = get_continuous_chunks(txt)
    results += continuous_chunked
    count.append(curcount)
    fp.close()
# record the finishing time and calculating the model processing time.
end = datetime.now()
print('time', (end - start).seconds)
# print the organizations and organization numbers
# print(results)
print(count)
# save the results for further research
pd.DataFrame(results).to_csv('NLTK_articles.csv', index=False, header=None)

results = pd.read_csv('NLTK_articles.csv', header=None).values
results = [x for [x] in results]

# reorganize the reults into map
map = {}
for org in results:
    if type(org) == type('a'):
        if org not in map:
            map[org] = 1
        else:
            map[org] += 1
print(map)

# remove alias
results = remove_alias(map)
names = [x[0] for x in results]
values = [x[1] for x in results]
df = pd.DataFrame()
df['Organizations'] = names
df['Counts'] = values
df.to_csv('NLTK_articles_alias.csv', index=False)

# rank the frequency and get the 5 highest ones
print(df.head(10))

# rough locate the word in the corpus, find the article name, sentense when doing analysing
org = 'ING'
content = []
for file in files:
    fp = open(path + '/' + file)
    txt = fp.read()
    l = len(sent_tokenize(txt))
    for sent in sent_tokenize(txt):
        if org in sent:
            print('file name:', file)
            print('sentense:', sent)
            content.append(file)
        # line.strip().split('/n')
        # if word in line:
        #     print(line)
    fp.close()
test = pd.DataFrame()
test['File'] = content
test.to_csv('test.csv', header=None)
print(test.head())

# precise locate the word in the corpus, find the article name, sentense line when doing analysing
phrase = 'S&P'
content = []
for file in files:
    fp = open(path + '/' + file)
    txt = fp.read()
    for i, sent in enumerate(sent_tokenize(txt)):
        for chunk in ne_chunk(pos_tag(word_tokenize(sent))):
            if hasattr(chunk, 'label') and chunk.label() == 'ORGANIZATION':
                org = ' '.join(c[0] for c in chunk)
                if org == phrase:
                    print('Organization:', org, 'File:', file, 'Sentence Number:', i)
                    print(sent)
                    content.append(file)
                # print the location and sentensence if neccessary
                # if org == phrase:
                #     print('Organization:', org, 'File:', file, 'Sentence Number:', i)
                #     print(sent)
    fp.close()
pd.DataFrame(content).to_csv('test.csv')

# rank the frequency and get the 5 highest ones
fdist = FreqDist(results)
frequents = [tag for (tag, _) in fdist.most_common(10)]

print(frequents)
for frequent in frequents:
    print(frequent, fdist.get(frequent))
tag = [tag for (tag, _) in fdist.most_common()]
num = [num for (_, num) in fdist.most_common()]
out = pd.DataFrame()
out['Organization'] = tag
out['Counts'] = num
out.to_csv('NLTK_ORG1.csv')
