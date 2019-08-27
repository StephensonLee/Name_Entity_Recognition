from nltk import FreqDist
from stanfordcorenlp import StanfordCoreNLP
from datetime import  datetime
import os
import pickle

nlp = StanfordCoreNLP(r'D:\360Downloads\stanford-corenlp-full-2018-02-27',lang='en')

def get_continuous_chunks(text):
    chunked = nlp.ner(text)
    continuous_chunk = []
    all_chunk = []
    current_chunk = []
    # print(chunked)
    for word,label in chunked:
        if label == 'ORGANIZATION':
            current_chunk.append(word)
        elif current_chunk:
            named_entity = " ".join(current_chunk)
            all_chunk.append(named_entity)
            if named_entity not in continuous_chunk:
                continuous_chunk.append(named_entity)
            current_chunk = []
        else:
            continue

    if current_chunk:
        named_entity = " ".join(current_chunk)
        all_chunk.append(named_entity)
        if current_chunk not in continuous_chunk:
            continuous_chunk.append(named_entity)
    return [all_chunk],[continuous_chunk]

path = "CCAT"
files= os.listdir(path)
results = []
start=datetime.now()
for file in files:
    # print(path+'/'+file)
    fp = open(path+'/'+file)
    txt = fp.read()
    # print(txt)
    all_chunked,continuous_chunked = get_continuous_chunks(txt)
    # print(all_chunked)
    # print(continuous_chunked)
    results+=all_chunked
    # if 'Income' in all_chunked:
    #     print(file)
end=datetime.now()
print('time',(end-start).seconds)
print(results)
nlp.close() # Do not forget to close! The backend server will consume a lot memery.

with open("test1.txt", "wb") as fp:
    pickle.dump(results, fp)
nlp.close()

with open("test1.txt", "rb") as fp:
    results = pickle.load(fp)
print(results)

# for article in text:
#     for i,word in enumerate(article):
#         results.append(word)
#         if i>0:
#             pairs.append([word,article[i-1]])
#         if i < len(article) - 1:
#             pairs.append([word,article[i+1]])
# print(pairs)
# print(results)
# fdist = FreqDist(results)
# # print([tag for (tag, _) in fdist.most_common()])
# frequents = [tag for (tag, _) in fdist.most_common()]
# print(frequents)
# for frequent in frequents:
#     print(frequent,fdist.get(frequent))
# freq = fdist.freq(common[0])+fdist.freq(common[1])+fdist.freq(common[2])
# print(freq)

corpus=results
words = []
for text in corpus:
    for word in text:
        words.append(word)
words = set(words)
print(words)
# Generate context for each word in a defined window size
word2int = {}
for i, word in enumerate(words):
    word2int[word] = i
# print('word2int',word2int)
# sentences = []
# for sentence in corpus:
#     sentences.append(sentence.split())
# print ('sentences', len(sentences),sentences)

WINDOW_SIZE = 1
data = []
for sentence in corpus:
    for idx, word in enumerate(sentence):
        for neighbor in sentence[max(idx - WINDOW_SIZE, 0): min(idx + WINDOW_SIZE, len(sentence)) + 1]:
            if neighbor != word:
                data.append([word, neighbor])
# print('data',len(data),data)

# Now create an input and a output label as reqired for a machine learning algorithm.
import pandas as pd
df = pd.DataFrame(data, columns=['input', 'label'])
# print(df)
# print(df.shape)
# print(word2int)

#Deinfine the tensor flow graph. That is define the NN
import tensorflow as tf
import numpy as np
ONE_HOT_DIM = len(words)
# function to convert numbers to one hot vectors
def to_one_hot_encoding(data_point_index):
    one_hot_encoding = np.zeros(ONE_HOT_DIM)
    one_hot_encoding[data_point_index] = 1
    return one_hot_encoding
X = [] # input word
Y = [] # target word
for x, y in zip(df['input'], df['label']):
    X.append(to_one_hot_encoding(word2int[ x ]))
    Y.append(to_one_hot_encoding(word2int[ y ]))

# convert them to numpy arrays
X_train = np.asarray(X)
Y_train = np.asarray(Y)

# making placeholders for X_train and Y_train
x = tf.placeholder(tf.float32, shape=(None, ONE_HOT_DIM))
y_label = tf.placeholder(tf.float32, shape=(None, ONE_HOT_DIM))

# word embedding will be 2 dimension for 2d visualization
EMBEDDING_DIM = 2

# hidden layer: which represents word vector eventually
W1 = tf.Variable(tf.random_normal([ONE_HOT_DIM, EMBEDDING_DIM]))
b1 = tf.Variable(tf.random_normal([1])) #bias
hidden_layer = tf.add(tf.matmul(x,W1), b1)

# output layer
W2 = tf.Variable(tf.random_normal([EMBEDDING_DIM, ONE_HOT_DIM]))
b2 = tf.Variable(tf.random_normal([1]))
prediction = tf.nn.softmax(tf.add( tf.matmul(hidden_layer, W2), b2))

# loss function: cross entropy
loss = tf.reduce_mean(-tf.reduce_sum(y_label * tf.log(prediction), axis=[1]))

# training operation
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

#Train the NN
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

iteration = 5000
for i in range(iteration):
    # input is X_train which is one hot encoded word
    # label is Y_train which is one hot encoded neighbor word
    sess.run(train_op, feed_dict={x: X_train, y_label: Y_train})
    if i % 3000 == 0:
        print('iteration '+str(i)+' loss is : ', sess.run(loss, feed_dict={x: X_train, y_label: Y_train}))

# Now the hidden layer (W1 + b1) is actually the word look up table
vectors = sess.run(W1 + b1)
print('vectors',vectors)

#Print the word vector in a table
w2v_df = pd.DataFrame(vectors, columns = ['x1', 'x2'])
w2v_df['word'] = words
w2v_df = w2v_df[['word', 'x1', 'x2']]
print('w2v_df',w2v_df)
w2v_df.to_csv("w2v_df(2).csv")


# Calculating neibours in 2 dimensions
w2v_df = pd.read_csv("w2v_df(2).csv")
words=w2v_df['word']
print(words)
w2w_dis = pd.DataFrame(words, columns = ['word'])
neibors=pd.DataFrame(words, columns = ['word'])
neib1=[]
neib2=[]
for word,x1,x2 in zip(w2v_df['word'], w2v_df['x1'], w2v_df['x2']):
    temp=[]
    for word_t, x1_t, x2_t in zip(w2v_df['word'], w2v_df['x1'], w2v_df['x2']):
        vector1 = np.array([x1,x2])
        vector2 = np.array([x1_t, x2_t])
        # dist = np.linalg.norm(vector1 - vector2)
        codist= np.dot(vector1, vector2) / (np.linalg.norm(vector1) * (np.linalg.norm(vector2)))
        temp.append(codist)
    # print(temp)
    w2w_dis[word]=temp
    temp1=temp.copy()
    temp1.sort(reverse=True)
    neib1.append(words[temp.index(temp1[1])])
    neib2.append(words[temp.index(temp1[2])])

neibors['neibour1']=neib1
neibors['neibour2']=neib2
neibors.to_csv("neibours(2).csv")
print(neibors)
print(w2w_dis)

# Now print the word vector as a 2d chart
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
for word, x1, x2 in zip(w2v_df['word'], w2v_df['x1'], w2v_df['x2']):
    ax.annotate(word, (x1, x2))
PADDING = 1.0
x_axis_min = np.amin(vectors, axis=0)[0] - PADDING
y_axis_min = np.amin(vectors, axis=0)[1] - PADDING
x_axis_max = np.amax(vectors, axis=0)[0] + PADDING
y_axis_max = np.amax(vectors, axis=0)[1] + PADDING
plt.xlim(x_axis_min, x_axis_max)
plt.ylim(y_axis_min, y_axis_max)
plt.rcParams["figure.figsize"] = (10, 10)
plt.show()