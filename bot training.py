import nltk
nltk.download('punkt')
from nltk.stem import WordNetLemmatizer
import json 
import pickle
import random
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout
from tensorflow.keras.optimizers import SGD

lemmatizer = WordNetLemmatizer()

words=[]
classes=[]
documents=[]
ignore_words=['?','!','.']
data_file=open('intents.json',errors="ignore").read()
intents=json.loads(data_file)

for intent in intents['intents']:
    for pattern in intent['patterns']:
       #tokenization here
       w=nltk.word_tokenize(pattern)
       words.extend(w)
       documents.append((w,intent['tag']))
       #add the tag to classes list
       if intent['tag']not in classes:
          classes.append(intent['tag'])

#lemmatizing words
words= [lemmatizer.lemmatize(w.lower())for w in words if w not in ignore_words]
words=list(set(words))
classes=list(set(classes))

#now lets convert this into a pickle file

pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))


#creating a training array and output array
training=[]
output_empty=[0]*len(classes)

#generating bag of words

for doc in documents:
  bag=[]
  pattern_words=doc[0]
  pattern_words=[lemmatizer.lemmatize(word.lower())for word in pattern_words]

  for w in words:
       bag.append(1) if w in pattern_words else bag.append(0)

  output_row=list(output_empty)
  output_row[classes.index(doc[1])]=1
  training.append([bag,output_row])

random.shuffle(training)

training=np.array(training,dtype=object)

train_x=list(training[:,0])
train_y=list(training[:,1])

#now lets just create a model
#our model is sequential

model=Sequential()
model.add(Dense(128,input_shape=(len(train_x[0]),),activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]),activation='softmax'))

# here we use SGD(stochastic gradient descent) optimizer function 

sgd=tf.keras.optimizers.legacy.SGD(learning_rate = 0.01,decay =1e-6,momentum=0.9,nesterov=True)
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

mfit=model.fit(np.array(train_x),np.array(train_y),epochs=200,batch_size=5,verbose=1)
model.save('chatbot_model.h5',mfit)

print("MODEL CREATED")