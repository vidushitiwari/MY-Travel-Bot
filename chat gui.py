import tkinter 
from tkinter import*
from keras.models import load_model
import nltk
nltk.download('punkt')
import tensorflow as tf
from nltk.stem import WordNetLemmatizer
import json
import pickle
import random
import numpy as np 
lemmatizer = WordNetLemmatizer()

intents=json.loads(open('intents.json',errors="ignore").read())
model=load_model('chatbot_model.h5',compile=True)
words=pickle.load(open('words.pkl','rb'))
classes=pickle.load(open('classes.pkl','rb'))

def bow(sentence):
	sentence_words=nltk.word_tokenize(sentence)
	sentence_words=[lemmatizer.lemmatize(word.lower())for word in sentence_words]
	bag=[0]*len(words)
	for s in sentence_words:
		for i,w in enumerate(words):
			if w==s:
				bag[i]=1
	return (np.array(bag))

def predict_class(sentence):
    sentence_bag=bow(sentence)
    #bow-->Bag Of Words
    res=model.predict(np.array([sentence_bag]))[0]
    ERROR_THRESHOLD=0.25
    results=[[i,r]for i,r in enumerate(res) if r>ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1],reverse=True)
    return_list=[]
    for r in results:
    	return_list.append({'intent':classes[r[0]],'probablity':str(r[1])})
    return return_list

def getResponse(ints):
    tag=ints[0]['intent']
    list_of_intents=intents['intents']
    for i in list_of_intents:
        if(i['tag']==tag):
           result=random.choice(i['responses'])
           break
    return result
    
def chatbot_response(msg):
    ints=predict_class(msg)
    res=getResponse(ints)
    return res

def send():
    msg=TextEntryBox.get("1.0",'end-1c').strip()
    TextEntryBox.delete('1.0','end')

    if msg!='':
         ChatHistory.config(state=NORMAL)
         ChatHistory.insert('end',"You: "+msg+"\n\n")    
    
         res=chatbot_response(msg)
         ChatHistory.insert('end',"Bot: "+res+"\n\n")  
         ChatHistory.config(state=DISABLED)
         ChatHistory.yview('end')
base= tkinter.Tk()
base.title("BOT  : )")
base.geometry("600x550")
base.resizable(width=True,height=True)





#chathistory text view
ChatHistory=Text(base,bd=0,bg='#78a4f5',font=('Cosmic Sans MS',15))
ChatHistory.config(state=DISABLED)

SendButton=Button(base,font=('Monotype Corsiva',20,'bold'),text="Send",bg="#4ccccf",activebackground="#89b6cc",fg="#000000",command=send)

TextEntryBox=Text(base,bd=0,bg='#e38fc6',font=('Open Sans',18,'bold'))

ChatHistory.place(x=6,y=6,height=500,width=700)
TextEntryBox.place(x=128,y=490,height=65,width=600)
SendButton.place(x=6,y=490,height=65,width=125)         
base.mainloop()























































