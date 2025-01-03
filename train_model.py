from __future__ import absolute_import, division, print_function, unicode_literals 
  
import numpy as np 
import tensorflow as tf 
from keras.models import Sequential 
from keras.layers import Dense, Activation 
from keras.layers import LSTM 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, LSTM

from keras.optimizers import RMSprop 
from datasets import load_dataset
from keras.callbacks import LambdaCallback 
from keras.callbacks import ModelCheckpoint 
from keras.callbacks import ReduceLROnPlateau
import pandas as pd 
import random 
import sys 
ds = load_dataset("Studeni/AMAZON-Products-2023")

dataset={'title':ds['train']['title'][:1000], 'description':ds['train']['description'][:1000]}

df=pd.DataFrame(dataset)
document=''
for a in df['title']:
    dodcument=document+' ' + a
for a in df['description']:
    document=document+' '+ a
vocabulary = sorted(list(set(document)))
char_to_indices = dict((c, i) for i, c in enumerate(vocabulary)) 
indices_to_char = dict((i, c) for i, c in enumerate(vocabulary)) 
X=[]
y=[]
for a in df['title']:
    l=[]
    for b in a:
        try:
            l.append(char_to_indices[b])
        except KeyError:
            l=l
    X.append(l)
for a in df['description']:
    l=[]
    for b in a:
        try:
            l.append(char_to_indices[b])
        except KeyError:
            l=l
    y.append(l)
i=0
X1=[]
y1=[]
while i<1000:
    if len(X[i])>30 and len(y[i])>400:
        X1.append(X[i][:30])
        y1.append(y[i][:400])
        i=i+1
    else:
        i=i+1
X1=np.array(X1)
y1=np.array(y1)
print(X1.shape)
print(y1.shape)
model = Sequential() 
model.add(LSTM(50, input_shape=(567, 30)))
model.add(Dense(1))
model.add(Activation('softmax')) 
model.compile(loss ='categorical_crossentropy', optimizer = 'adam') 
model.fit(X1, y1, batch_size = 128, epochs = 500) 
