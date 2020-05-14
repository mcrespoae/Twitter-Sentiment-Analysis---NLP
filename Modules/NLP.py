# -*- coding: utf-8 -*-
"""
Created on Sun May  3 10:27:07 2020

@author: Mario Crespo
"""
#Sklearn
from sklearn.metrics import confusion_matrix

#Utility
import time




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
            if classifier == "All": pass #Don't use this model since it requires a dense array to test it.
            elif classifier == "GaussianNB":
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
    
