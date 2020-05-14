# -*- coding: utf-8 -*-
"""
Created on Sun May  3 10:27:07 2020

@author: Mario Crespo
"""
#Custom
from Modules import Menu
from Modules import NLP

#Dataframe
import pandas as pd
import numpy as np

#Sklearn
from sklearn.model_selection import train_test_split


#------------------Testing Function--------------#

def testing(skipPreprocess, exportPreprocessDataToFile, model):
    #Try the model with your own sentence
    print("\n-------- TESTING--------")
    #Write some sentences to test them
    testText = ["I am too happpppy @Mario #NLP xD ;)","I wasn't háppy!! :( 66 www.mariocrespo.es"]
    dfTest = pd.DataFrame(testText, columns=['Text'])
    
    if skipPreprocess==0:
        inputTestPreprocessed = dfTest['Text'].apply(Preprocess.textPreprocessing)
        if exportPreprocessDataToFile == 1:
            inputTestPreprocessed.to_csv('./Test/PreprocessedTestText.csv',index=False, encoding = "utf-8")
        
    elif skipPreprocess==1:
        #Load the dataframe
        inputTestPreprocessedDF = pd.read_csv('./Test/PreprocessedTestText.csv', encoding = "utf-8")
        inputTestPreprocessed=inputTestPreprocessedDF.Text
        
    inputTextVectorized = vectorizationApproach.transform(inputTestPreprocessed)
    y_pred = model.predict(inputTextVectorized)
    y_pred = np.where(y_pred==4,"Positive","Negative")        
    
    print("Original text\t\t\t\t\t                    Preprocessed text\t\t\t\t               Prediction\n"
          +str(testText[0])+ "\t\t\t\t    "+str(inputTestPreprocessed[0])+"\t\t            "+str(y_pred[0])+"\n"
          +str(testText[1])+ "\t\t    "+str(inputTestPreprocessed[1])+"\t\t\t                        "+str(y_pred[1])) 

#------------------Main------------------#
if __name__ == "__main__":
    #Show the start menu
    skipPreprocess,fileToImport,fileToExport,dataFrameUsage,exportPreprocessDataToFile=Menu.startMenu()
    
    #Select vectorization apporach and classifier
    approach,classifier=Menu.approachAndClassifierMenu()
    
    if skipPreprocess==0:
        #Import the preprocess file and call the function to preprocess the data
        from Modules import Preprocess
        tweetTextPreprocessed, y = Preprocess.Preprocessing(dataFrameUsage, exportPreprocessDataToFile, fileToExport)
            
    elif skipPreprocess==1:
        #Load the dataframe already preprocessed
        print("Reading "+str(fileToImport)+" file with preprocessed data.")
        preprocessedData = pd.read_csv(fileToImport, encoding = "utf-8")
        y=preprocessedData.Target
        tweetTextPreprocessed=preprocessedData.Text
    
    #Vectorize the data
    print("Using "+str(approach)+" approach.")
    vectorizationApproach = NLP.vectorization(approach)
    X = vectorizationApproach.fit_transform(tweetTextPreprocessed.values.astype('U'))#The 'U' is for unicode since when reading the preprocessed .csv may return some non unicode values
    
    #Split the data. The data has been shuffled before, so we don't need to do it again but we keep it just in case we want to run the whole dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
    
    # Train and predict
    print("Using "+str(classifier)+" classifier(s)")
    model = NLP.trainClassifierandPrint(X_train, X_test, y_train, y_test, classifier=classifier)
    
    #Function to test the preprocessing, vectorization and model features
    testing(skipPreprocess, exportPreprocessDataToFile, model)
