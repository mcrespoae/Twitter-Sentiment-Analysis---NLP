# -*- coding: utf-8 -*-
"""
Created on Sun May  3 10:27:07 2020

@author: Mario Crespo
"""
#Sklearn
from sklearn.metrics import confusion_matrix

#Utility
import time

#Dataframe and Numpy
import numpy as np
import pandas as pd





#------------------NLP--------------#
def W2Vvectorization(tweetTextPreprocessed, sizeW2v=50):
    from gensim.models import Word2Vec
    #We create a list with the text
    rawList=tweetTextPreprocessed.values.astype('U').tolist()
    #Create a list of list. Each document will be a list
    sentences = [row.split(',') for row in rawList]
    
    #Create a list where each token will be an element of a list
    w2vlist=[]
    for sentence in sentences:
        for token in sentence:
            sub=token.split(' ')
        w2vlist.append(sub)
    
    #Create the Word2VecModel
    vector=Word2Vec(w2vlist, min_count=10, size=sizeW2v, workers=7, window=3, sg=1, sample=0.01)   
    w2vDict = dict(zip(vector.wv.index2word, vector.wv.vectors))
    return w2vDict, w2vlist #Return the dictionary with the tokens and its numerical vectors and the final list
    
def W2VGetDataFrameFromDict(w2v, w2vlist, sizeW2v):
    #Create a list with the mean of the values of each document
    tempList=[]
    for sentence in w2vlist:
        values=0
        for token in sentence:
            if token in w2v: #If the token is inside the dict, sum it
                values=w2v[token]+values
            else:#If it is not inside the dict, fill it with 0
                values=np.zeros(sizeW2v)+values

        tempList.append(values/sizeW2v)#Calculate the mean of each column of each document and add it to a list
    
    Xw2v=pd.DataFrame(tempList)#Create a DataFrame with the values of the Word2Vec model
    return Xw2v

def W2VTestData(w2vDict, inputTestPreprocessed, sizeW2v):
    #Create a list of list. Each document will be a list
    sentences = [row.split(',') for row in inputTestPreprocessed]
    
    #Create a list where each token will be an element of a list
    w2vlist=[]
    for sentence in sentences:
        for token in sentence:
            sub=token.split(' ')
        w2vlist.append(sub)
      
    X = W2VGetDataFrameFromDict(w2vDict,w2vlist,sizeW2v)
    return X
   
def vectorization(approach="TFIDF"): 
    if approach == "TFIDF":
        # Create the matrix with TfidfVectorizer from our already tokenized text
        from sklearn.feature_extraction.text import TfidfVectorizer
        return TfidfVectorizer(sublinear_tf=True, min_df=0.005, max_df=0.8)

    
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
        try:
            tempModelScore = createModel("Multinomial Naive Bayes", tempModel, X_train, X_test, y_train, y_test)
            if tempModelScore > modelScore:
                model=tempModel
                modelScore=tempModelScore
        except:
            if classifier == "All":
                print("\nCouldn't use MultinomialNB since it requires only possitive values. Skipping this classifier..")
            elif classifier == "MultinomialNB":
                print("\nCouldn't use MultinomialNB since it requires only possitive values. Switching to Logistic Regression classifier..." )
                model = trainClassifierandPrint(X_train, X_test, y_train, y_test, classifier="Logistic Regression")
                return model
    
    if classifier == "GaussianNB" or classifier == "All":
        notFound=1         
        from sklearn.naive_bayes import GaussianNB
        tempModel = GaussianNB()
        #Too much data will generate an error. 
        #We avoid this by using try/except
        #We also don't use this model to predict since it needs an sparse array
        try:
            #GaussianNB doens't work with sparse array so we change it to dense
            X_trainGNB = X_train.toarray() 
            X_testGNB = X_test.toarray()
            tempModelScore=createModel("Gaussian Naive Bayes", tempModel, X_trainGNB, X_testGNB, y_train, y_test)
        except:
            if classifier == "All":
                print("\nSkipping Gaussian Naive Bayes due to memory restrictions.")#Skip this model
        if classifier == "GaussianNB":
            print("\nCannot use this model to predict. Switching to Logistic Regression classifier...")
            model = trainClassifierandPrint(X_train, X_test, y_train, y_test, classifier="Logistic Regression")
            return model
   
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
        tempModel = SVC(kernel="linear",gamma='auto')
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
    
