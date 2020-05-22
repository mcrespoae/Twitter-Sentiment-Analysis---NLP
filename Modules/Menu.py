# -*- coding: utf-8 -*-
"""
Created on Sun May  3 10:27:07 2020

@author: Mario Crespo
"""

#Utility
from os import listdir, getcwd, path
import sys 

#------------------StartMenu--------------#
def exportData(dataFrameUsage):
    exportDataBoolean=False
    path=""
    while exportDataBoolean==False:
        dataReceived=str(input("Do you want to store the preprocessed data to a file? (y/n): "))
        if dataReceived.lower() == 'y':#If yes, we return the path and the name of the file
            path=getcwd()
            path=path+'\PreprocessedTweets'+'\\'+'PreprocessedTweets'+str(int(dataFrameUsage*100))+'.csv'
            return(True, path)
       
        elif dataReceived.lower() == 'n': return(False, path)
                
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
        print("0) To abort the program")
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
        
        dataReceived=str(input("Do you want to skip the preprocessing by loading a preprocessed file? (y/n): "))
        
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


#------------------Classifier and Vector Menu selection--------------#
def approachAndClassifierMenu():
    approachList=["Bag of Words","TFIDF", "Word2Vec"]
    classifierList=["Logistic Regression", "MultinomialNB", "GaussianNB", "KNN", "SVC", "RandomForest","NN", "All"]
    print("Which vectorization approach do you want to use?")
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