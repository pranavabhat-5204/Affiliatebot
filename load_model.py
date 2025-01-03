data= {'title':ds['title'],'description':ds['title']}
df=pd.DataFrame(data)
tokenized_dataset1=dataset['title'].map()
model = Sequential() 
model.add(LSTM(128, input_shape =(max_length, len(vocabulary)))) 
model.add(Dense(len(vocabulary))) 
model.add(Activation('softmax')) 
optimizer = RMSprop(lr = 0.01) 
model.compile(loss ='categorical_crossentropy', optimizer = optimizer) 

model = Sequential() 
model.add(LSTM(128, input_shape =(567, 30)))
model.add(Dense(len(vocabulary))) 
model.add(Activation('softmax')) 
model.compile(loss ='categorical_crossentropy', optimizer = 'adam') 

model.fit(X1, y1, batch_size = 128, epochs = 500) 
model.save('gfgModel.h5')
X1=X1.reshape(None,567,30)
