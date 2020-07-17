# -*- coding: utf-8 -*-
"""
Created on Sun May  3 10:27:07 2020

@author: Mario Crespo
"""
#Utility
import time

#Multiprocessing
import concurrent.futures

#Dataframe
import pandas as pd

#spaCy
import en_core_web_md

#Import Utilities
from pycontractions import Contractions
import unidecode
from bs4 import BeautifulSoup
import re
from datetime import datetime, timedelta




#------------------Preprocessing--------------#

def prepareData(filename, colNames, header=0, encoding ='utf8', dataFrameUsage=0.002):
    
    data = pd.read_csv(filename, header=header, names=colNames, encoding=encoding)
    # Get some shuffle data since it cannot open all the data at once.
    df = pd.DataFrame(data).sample(frac=dataFrameUsage, random_state=1) 
    
    #Get the text and class data
    df = df.drop(columns=['Id', 'Date', 'Flag', 'User'])
    df=df.dropna()#Delete the NaN rows

    X=df.drop(columns=['Target'])
    y=df.Target

    return X,y
   
    
def textPreprocessing(text):
 
    #First remove the HTML tags just in case
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text(separator=" ") 
    
    text = re.sub('[\s]+', ' ', text)#Remove extra whitespaces
    
    text = re.sub('@[^\s]+','atusername', text)#Replace @username for atusername
    
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','',text)#Remove links
    
    text = re.sub('([a-zA-Z])\\1{2,}', "\\1\\1", text)#Remove characters than are repeated 3 or more times
    
    #Faces like :) :( xD :'( (: etc
    text = re.sub('(:|;)-?(\)|D|p|P)', "happy", text)
    text = re.sub('\(-?(:|;)', "happy", text)
    text = re.sub(':\'?-?\(', "sad", text)
    text = re.sub('\)-?\'?:', "sad", text)
    text = re.sub('(x|X)D', "fun", text)
    
    text = re.sub(r'#([^\s]+)', r'\1', text)#Remove the # from the #topic
    
    text = unidecode.unidecode(text)#Remove accents   
    
    #Sometimes it gets stuck here
    try:
        text = list(cont.expand_texts([text], precise=True))[0]#Expand contractions
    except:
        text=str(text)

    text = text.lower()#Convert the text to lowercase
    
    #Now tokenize the text
    doc = nlp(text)
    cleanedText=[]
       
    for token in doc:
        if token.pos_ == 'PUNCT': pass #Delete punctuation
        elif token.pos_ == 'NUM' or token.text.isnumeric(): pass #Delete numbers        
        elif token.is_stop: pass #Delete the stopwords
        elif token.pos_ == 'SYM': pass #Delete special characters 
        elif token.lemma_ != '-PRON-' and token.lemma_ != "" and token.lemma_ != " ":
            cleanedText.append(token.lemma_)#If we are here is because this token lemma has to be added to the list      
    return " ".join(cleanedText)#Returns a string

#Preprocess each tweet of each chunnk and returns a list
def textPreprocessingChunk(chunks):
    preprocessedText=[]
    for chunk in chunks:
        preprocessedText.append(textPreprocessing(chunk))
    return preprocessedText                           

def Preprocessing(dataFrameUsage, exportPreprocessDataToFile, fileToExport):
    #Open the file and do a fist general pass to preprocess the data
    print("Loading a " +str(dataFrameUsage*100)+ "% of the data.")
    tweetText, y = prepareData(fileName, colNames, encoding=encoding, dataFrameUsage=dataFrameUsage)
    
    #Preprocess the data specifically for NLP
    preprocessTime=0.0085*len(tweetText.index) #Get the estimate preprocess time in seconds 
    print("Preprocessing the text. It will take up to "+str(time.strftime("%Hh:%Mm:%Ss.", time.gmtime(preprocessTime)))+
          " It should be done at: " + (datetime.now() + timedelta(seconds=preprocessTime)).strftime('%H:%M:%S.'))

    startPreprocessingTime=time.time()
    
    #Divide the whole dataset into n chunks. Each chunk will be a thread.
    n=20
    chunks = [tweetText['Text'][i * n:(i + 1) * n] for i in range((len(tweetText['Text']) + n - 1) // n )]
              
    #Preprocess the text in different threads
    with concurrent.futures.ProcessPoolExecutor() as executor:
        #Create one thread per each chunk and returns it to the results object generator.
        #It will finish only when all the threads are finished
        results = executor.map(textPreprocessingChunk, chunks) 
        
        #Convert the generator object into list
        tweetTextPreprocessedLines=[]
        for result in results:
            for tweet in result:
                tweetTextPreprocessedLines.append(tweet)
  
    #Stores it in a Series to be recovered or stored later
    tweetTextPreprocessed = pd.Series(tweetTextPreprocessedLines, index=tweetText.index, name='Text')


    print("The real time spent in the preprocessing stage was: "+
          str(time.strftime("%Hh:%Mm:%Ss.", time.gmtime(time.time()-startPreprocessingTime))))
    
    
    #Store in a file the preprocessed tweetd
    if exportPreprocessDataToFile == 1:
        dfExport=pd.concat([tweetTextPreprocessed, y], axis=1)
        dfExport.to_csv(fileToExport, index=False, encoding = "utf-8")
        
    return tweetTextPreprocessed, y

#--------------------------Main--------------------------#

if __name__ == "Modules.Preprocess":
    #Setting up the dataset
    colNames=['Target','Id','Date','Flag','User','Text'] #Get the name of the columns
    encoding="ISO-8859-1"#utf8 cannot read some special characters so ISO-8859-1 is used
    fileName='tweet_dataset.csv'
    
    
    print("Loading language model. This may take several seconds.")
    nlp = en_core_web_md.load()#Load the English medium spaCy language model('en_core_web_md')
    
    print("Loading GloveTwitter model. This may take up to 1 minute.")
    # Choose the model. Others such us "word2vec-google-news-300"" are available too.
    #Use "glove-twitter-100" (<1GB) or "glove-twitter-200" (1GB)for final results. "glove-twitter-25"(200MB) is just for fast checks
    cont = Contractions(api_key="glove-twitter-100")
    cont.load_models()    #Get the contractions for English and prevents loading on firs expand_text call
    
    #Exlude some words with potential negative sentimental analysis
    deselect_stop_words = ["no", "not", "n't", "less", "enough", "never"]
    for w in deselect_stop_words:
            nlp.vocab[w].is_stop = False