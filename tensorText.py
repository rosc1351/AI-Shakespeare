import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
import copy

text = open('Shake.txt','rb').read().decode(encoding='utf-8')
vocab = sorted(set(text))

char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)
text_as_int = np.array([char2idx[c] for c in text])

Iwidth = 32

def str2Img(s,vocabSize):
    x = np.zeros((vocabSize,len(s)))
    for j in range(len(s)):
            c = char2idx[s[j]]
            x[c,j] = 1
    return x

def genNTextImg(n, intText,vocabSize, imgWidth):
    data = []
    nextL = []
    textLen = len(intText)
    dSample = np.floor((textLen - imgWidth - 1) / n)
    for i in range(n):
        x = np.zeros((vocabSize, imgWidth))
        for j in range(imgWidth):
            c = intText[int(i*dSample+j)]
            x[c,j] = 1

        c = intText[int(i*dSample + imgWidth)]
        data.append(np.array(x))
        nextL.append(c)
    return np.array(data),np.array(nextL)

def generateText(seed, imgWidth, length):
    outText = ""
    if len(seed) < imgWidth:
        seed = (" "*(imgWidth-len(seed))) + seed
    seed = seed[:imgWidth]
    for i in range(length):
        x = str2Img(seed,len(vocab))
        x = np.array([x])
        y = np.argmax(model.predict(x))
        y = idx2char[y]

        outText = outText + y
        seed = seed[1:] + y
    print(outText)
    return outText

trainTxt,nextL = genNTextImg(10000,text_as_int,len(vocab),Iwidth)

print(trainTxt.shape)
print(nextL.shape)

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(len(vocab), Iwidth)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(len(vocab), activation=tf.nn.softmax)
])

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

for i in range(1000):
    print(i)
    trainTxt,nextL = genNTextImg(50000,text_as_int[i:],len(vocab),Iwidth)
    model.fit(trainTxt, nextL, epochs=3)
    if i % 1 == 0:
        generateText("I need a really long sample text to start off the seed. If its not long enough, my code will fail, trust me, I've tried.", Iwidth, 250)

#It'll be a while before you get here