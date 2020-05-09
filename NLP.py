# -*- coding: utf-8 -*-
"""
Created on Sun May  3 10:27:07 2020

@author: Mario Crespo
"""

#Dataframe
import pandas as pd
import numpy as np

#Sklearn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

#Utility
import time
from os import listdir, getcwd, path
import sys 

#------------------Functions--------------#


#------------------Classifier and Vector Menu selection--------------#
def approachAndClassifierMenu():
    approachList=["Bag of Words","TFIDF"]
    classifierList=["Logistic Regression", "MultinomialNB", "GaussianNB", "KNN", "SVC", "RandomForest","NN", "All"]
    print("Which vectrization approach do you want to use?")
    notSelected=True
    while notSelected==True:
        for i, element in enumerate(approachList):
            print(str(i+1)+")",str(element)+" approach")
        
        selected=input("Choose: ")
        print("------")
        if selected.isdigit()==False: pass
        elif int(selected)>=1 and int(selected)<=len(approachList):
            notSelected=False
            approach=approachList[int(selected)-1]
        else: pass
    
    print("Which classifier do you want to use?")
    notSelected=True
    while notSelected==True:
        for i, element in enumerate(classifierList):
            print(str(i+1)+")",str(element)+" classifier.")
        
        selected=input("Choose: ")
        print("------")
        if selected.isdigit()==False: pass
        elif int(selected)>=1 and int(selected)<=len(classifierList):
            notSelected=False
            classifier=classifierList[int(selected)-1]
        else: pass
    return approach, classifier


#------------------StartMenu--------------#
def exportData(dataFrameUsage):
    exportDataBoolean=False
    while exportDataBoolean==False:
        dataReceived=str(input("Do you want to store the preprocessed data to a file? (y/n): "))
        if dataReceived.lower() == 'y':#If yes, we return the path and the name of the file
            path=getcwd()
            path=path+'\PreprocessedTweets'+'\\'+'PreprocessedTweets'+str(int(dataFrameUsage*100))+'.csv'
            return(True, path)
       
        elif dataReceived.lower() == 'n': return
                
        else: print("Please, enter 'y' or 'n'")


def percentageOfData():
    percentageOfDataBoolean=False
    while percentageOfDataBoolean==False:
        selected=input("How much % of data do you want to use?\nAttention, high values may take hours to preprocess. Use a 2 for quick demonstration.\nEnter a number between 1 and 100: ")
        print("------")
        if selected.isdigit()==False: pass
        elif int(selected)>=1 and int(selected)<=100:
            dataFrameUsage=float(int(selected)/100)
            return dataFrameUsage
        else: pass
        


def chooseFile():
    directory="PreprocessedTweets"
    fileChosen=False
    
    while fileChosen==False:
        print("Choose a preprocessed file:")
        print("0) To abort the programm")
        selected=False
        i=1
        filesFound=[]
        for f in listdir(directory):
            if f.endswith('.csv') and f.startswith('PreprocessedTweets'):
                # Extract digit string  
                numbers = ''.join(filter(lambda i: i.isdigit(), f)) 
                print(str(i)+")",str(f)+"\t\t---With a "+numbers+"% of the dataset.")
                i+=1
                filesFound.append(path.abspath(directory+'\\'+f))
        if i == 1:
            print("No preprocessed files found: changing to preprocess the data.")
            return "NotFound"
        
        selected=input("Enter a number: ")
        print("------")
        
        if selected.isdigit()==False: pass
        elif int(selected) == 0: sys.exit()
        elif int(selected) > 0 and int(selected) < i:
            fileChosen=True
            fileToImport=filesFound[int(selected)-1]
            return fileToImport
            
    return fileToImport


def startMenu():
    skipPreprocess=False
    exportPreprocessDataToFile=False
    dataFrameUsage=0.02
    fileToExport=""
    fileToImport=""
    
    skipPreprocessBoolean=False
    
    while skipPreprocessBoolean==False:
        
        dataReceived=str(input("Do you want to skip the preprocessing process by loading a preprocessed file? (y/n): "))
        
        if dataReceived.lower() == 'y':#If yes, we try to open the file
            fileToImport=chooseFile()
            if fileToImport=="NotFound":#If we couldn't find any file, go to preprocessData
                skipPreprocess=False
                dataFrameUsage=percentageOfData()
                exportPreprocessDataToFile, fileToExport=exportData(dataFrameUsage)
                return(skipPreprocess,fileToImport,fileToExport,dataFrameUsage,exportPreprocessDataToFile)
                
            skipPreprocess=True
            return(skipPreprocess,fileToImport,fileToExport,dataFrameUsage,exportPreprocessDataToFile)
        
        elif dataReceived.lower() == 'n':
            skipPreprocess=False
            dataFrameUsage=percentageOfData()
            exportPreprocessDataToFile, fileToExport =exportData(dataFrameUsage)
            return(skipPreprocess,fileToImport,fileToExport,dataFrameUsage,exportPreprocessDataToFile)
                
        else:
            print("Please, enter 'y' or 'n'")
            print("------")


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
    
    text = list(cont.expand_texts([text], precise=True))[0]#Expand contractions
       
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


#------------------NLP--------------#

def vectorization(approach="TFIDF"): 
    if approach == "TFIDF":
        # Create the matrix with TfidfVectorizer from our already tokenized text
        from sklearn.feature_extraction.text import TfidfVectorizer
        return TfidfVectorizer(sublinear_tf=True, max_df=0.8)

    
    elif approach == "Bag of Words":
         # Create the matrix with BOW from our already tokenized text
        from sklearn.feature_extraction.text import CountVectorizer
        return CountVectorizer(max_df=0.8)
    
    else:
        print(str(approach)+"  not found, switching to tfidf...")
        vector = vectorization(approach="TFIDF")
        return vector
        
def printingResults(classifier, modelScore, confusionMatrix, fitTime, predTime):
    
    print("\n-------- Results for " + classifier + " classifier--------")
    print("Confusion matrix: ")
    print(  "TN: " + str(confusionMatrix[0][0]) + "   FP: "+ str(confusionMatrix[0][1]) + 
          "\nFN: " + str(confusionMatrix[1][0]) + "   TP: "+ str(confusionMatrix[1][1]))
    print("\nScore: "+"{:.2f}".format(modelScore*100)+"%")
    print("\nFit Time: "+"{:.4f}".format(fitTime)+"s")
    print("Predict Time: "+"{:.4f}".format(predTime)+"s")
    print("--------  End of the report  --------")
    
def createModel(classifier, model, X_train, X_test, y_train, y_test):
    timeStart = time.time()
    model.fit(X_train, y_train)#Fit the model
    timeEnd= time.time()
    fitTime = timeEnd - timeStart
    
    modelScore = model.score(X_test, y_test)#Get the score of the model
    
    timeStart = time.time()
    y_pred = model.predict(X_test)#Predict the model
    timeEnd= time.time()
    predTime = timeEnd - timeStart
    
    confusionMatrix=confusion_matrix(y_test, y_pred) #Get the confusion Matrix
    
    #Print the results
    printingResults(classifier, modelScore, confusionMatrix, fitTime, predTime)
    
    return modelScore

def trainClassifierandPrint(X_train, X_test, y_train, y_test, classifier="All"):
    modelScore = 0
    notFound=0
    if classifier == "Logistic Regression" or classifier == "All":
        notFound=1 
        from sklearn.linear_model import LogisticRegression
        tempModel = LogisticRegression(max_iter=10000)
        #Call the function to fit, predict and get results
        tempModelScore = createModel("Logistic Regression", tempModel, X_train, X_test, y_train, y_test)
        if tempModelScore > modelScore:
            model=tempModel
            modelScore=tempModelScore
       
    if classifier == "MultinomialNB" or classifier == "All":
        notFound=1
        from sklearn.naive_bayes import MultinomialNB
        tempModel = MultinomialNB()
        tempModelScore = createModel("Multinomial Naive Bayes", tempModel, X_train, X_test, y_train, y_test)
        if tempModelScore > modelScore:
            model=tempModel
            modelScore=tempModelScore
    
    if classifier == "GaussianNB" or classifier == "All":
        notFound=1
        maxData=40000 #Threshold used to avoid memory issues
        #Too much data will generate an error. We avoid this by checking the size of the data and skipping this model if there is too much data
        countRow = y_train.size
        if classifier == "GaussianNB" and countRow > maxData:
            print("\nThere is too much data to create a model with GaussianNB. Switching to Logistic Regression classifier...")
            model = trainClassifierandPrint(X_train, X_test, y_train, y_test, classifier="Logistic Regression")
            return model
        
        elif classifier == "All" and countRow > maxData:
            print("\nThere is too much data to create a model with GaussianNB. Skipping this classifier.")
        else:
            from sklearn.naive_bayes import GaussianNB
            tempModel = GaussianNB()
            #GaussianNB doens't work with sparse array so we change it to dense
            X_trainGNB = X_train.toarray() 
            X_testGNB = X_test.toarray()
            tempModelScore=createModel("Gaussian Naive Bayes", tempModel, X_trainGNB, X_testGNB, y_train, y_test)
            if tempModelScore > modelScore:
                model=tempModel
                modelScore=tempModelScore
        
    if classifier == "KNN" or classifier == "All":
        notFound=1
        from sklearn.neighbors import KNeighborsClassifier
        tempModel = KNeighborsClassifier(n_neighbors=5)
        tempModelScore=createModel("KNN", tempModel, X_train, X_test, y_train, y_test)
        if tempModelScore > modelScore:
            model=tempModel
            modelScore=tempModelScore

    if classifier == "SVC" or classifier == "All":
        notFound=1
        from sklearn.svm import SVC
        tempModel = SVC(gamma='scale')
        tempModelScore=createModel("SVC", tempModel, X_train, X_test, y_train, y_test)
        if tempModelScore > modelScore:
            model=tempModel
            modelScore=tempModelScore
            
    if classifier == "RandomForest" or classifier == "All":
        notFound=1
        from sklearn.ensemble import RandomForestClassifier
        tempModel = RandomForestClassifier()
        tempModelScore=createModel("Random Forest", tempModel, X_train, X_test, y_train, y_test)
        if tempModelScore > modelScore:
            model=tempModel
            modelScore=tempModelScore  
            
    if classifier == "NN" or classifier == "All":
        notFound=1 
        from sklearn.neural_network import MLPClassifier
        tempModel = MLPClassifier(solver='adam',hidden_layer_sizes=(8,6),max_iter=1200)
        tempModelScore = createModel("Neural Network", tempModel, X_train, X_test, y_train, y_test)
        if tempModelScore > modelScore:
            model=tempModel
            modelScore=tempModelScore
            
    if notFound == 0:
        print(str(classifier)+" classifier not found, switching to Logistic Regression classifier...")
        model = trainClassifierandPrint(X_train, X_test, y_train, y_test, classifier="Logistic Regression")
        return model
         
    return model #Returns the best model
    

#------------------Main------------------#

#Show the start menu
skipPreprocess,fileToImport,fileToExport,dataFrameUsage,exportPreprocessDataToFile=startMenu()

#Select vectorization apporach and classifier
approach,classifier=approachAndClassifierMenu()



if skipPreprocess==0:
    #spaCy
    import en_core_web_md
    
    #Import Utilities
    from pycontractions import Contractions
    import unidecode
    from bs4 import BeautifulSoup
    import re
    from datetime import datetime, timedelta
    
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
       
    #Open the file and do a fist general pass to preprocess the data
    print("Loading a " +str(dataFrameUsage*100)+ "% of the data.")
    tweetText, y = prepareData(fileName, colNames, encoding=encoding, dataFrameUsage=dataFrameUsage)
    
    #Preprocess the data specifically for NLP
    preprocessTime=0.019*len(tweetText.index) #Get the estimate preprocess time in seconds 
    print("Preprocessing the text. It will take approximately "+str(time.strftime("%Hh:%Mm:%Ss.", time.gmtime(preprocessTime)))+
          " It should be done at: " + (datetime.now() + timedelta(seconds=preprocessTime)).strftime('%H:%M:%S.'))
    
    startPreprocessingTime=time.time()
    tweetTextPreprocessed = tweetText['Text'].apply(textPreprocessing)
    print("The real time spent in the preprocessing stage was: "+
          str(time.strftime("%Hh:%Mm:%Ss.", time.gmtime(time.time()-startPreprocessingTime))))


    #Store in a file the preprocessed tweetd
    if exportPreprocessDataToFile == 1:
        dfExport=pd.concat([tweetTextPreprocessed, y], axis=1)
        dfExport.to_csv(fileToExport, index=False, encoding = "utf-8")

elif skipPreprocess==1:
    #Load the dataframe already preprocessed
    print("Reading "+str(fileToImport)+" file with preprocessed data.")
    preprocessedData = pd.read_csv(fileToImport, encoding = "utf-8")
    y=preprocessedData.Target
    tweetTextPreprocessed=preprocessedData.Text

#Vectorize the data
print("Using "+str(approach)+" approach.")
vectorizationApproach = vectorization(approach)
X = vectorizationApproach.fit_transform(tweetTextPreprocessed.values.astype('U'))

#Split the data. The data has been shuffled before, so we don't need to do it again but we keep it just in case we want to run the whole dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)


# Train and predict
print("Using "+str(classifier)+" classifier(s)")
model = trainClassifierandPrint(X_train, X_test, y_train, y_test, classifier=classifier)

#Try the model with your own sentence
print("\n-------- TESTING--------")
#Write some sentences to test them
testText = ["I am too happpppy @Mario #NLP xD ;)","I wasn't háppy!! :( 66 www.mariocrespo.es"]
dfTest = pd.DataFrame(testText, columns=['Text'])

if skipPreprocess==0:
    inputTestPreprocessed = dfTest['Text'].apply(textPreprocessing)
    if exportPreprocessDataToFile == 1:
        inputTestPreprocessed.to_csv('PreprocessedTestText.csv',index=False, encoding = "utf-8")
    
elif skipPreprocess==1:
    #Load the dataframe
    inputTestPreprocessedDF = pd.read_csv('PreprocessedTestText.csv', encoding = "utf-8")
    inputTestPreprocessed=inputTestPreprocessedDF.Text
    
inputTextVectorized = vectorizationApproach.transform(inputTestPreprocessed)
y_pred = model.predict(inputTextVectorized)
y_pred = np.where(y_pred==4,"Positive","Negative")        

print("Original text\t\t\t\t\t                    Preprocessed text\t\t\t\t               Prediction\n"
      +str(testText[0])+ "\t\t\t\t    "+str(inputTestPreprocessed[0])+"\t\t            "+str(y_pred[0])+"\n"
      +str(testText[1])+ "\t\t    "+str(inputTestPreprocessed[1])+"\t\t\t                        "+str(y_pred[1]))
