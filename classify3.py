# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import time
import warnings
warnings.filterwarnings("ignore")


#RESULT EVALUATION PROCEDURE PROCEDURE
def analyzeResults(cm,classifier_name):
    # evaluate the performance of SLR
    TN = cm[0,0]
    FP = cm[0,1]
    FN = cm[1,0]
    TP = cm[1,1]
    accuracy = float(TP+TN)/(TP+FP+FN+TN)
    precision = float(TP)/(TP+FP)
    recall = float(TP)/(TP+FN)
    F1err = 2*( (precision*recall) / (precision+recall) )
    print("------------------oo---------------------------")
    print("RESULTS (" + classifier_name + ")")
    print("------------------oo---------------------------")
    print("Precision: " + str(precision*100) + "%")
    print("Recall:    " + str(recall*100) + "%")
    print("Accuracy:  " + str(accuracy*100) + "%")
    print("F1-Measure " + str(F1err*100) + "%")
    #print("Confusion Matrix: ")
    #print(cm)
    print("------------------oo---------------------------")
    print("             xxxxx||xxxxx                      ")
    print("")
    my_df = pd.DataFrame(cm)
    file_name = "Confusion_Mat_"+classifier_name+".csv"
    my_df.to_csv(file_name, index=False, header=False)
    return precision, recall, F1err, accuracy


X_train_main = X_train
y_train_main = y_train

#index of random seeds fot train/test split
randoms = np.array([42,7,13,23,18,39,31,27,11,35])
#split size
split_size = '80_20'
#create empty dataframes
raw_data = {'random_seed' : [0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'precision': [0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'recall' : [0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'f1err' : [0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'accuracy' : [0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'delay' : [0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}
KNN_results = pd.DataFrame(raw_data, columns = ['random_seed', 'precision', 'recall', 'f1err', 'accuracy', 'delay'])
raw_data = {'random_seed' : [0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'precision': [0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'recall' : [0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'f1err' : [0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'accuracy' : [0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'delay' : [0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}
NB_results = pd.DataFrame(raw_data, columns = ['random_seed', 'precision', 'recall', 'f1err', 'accuracy', 'delay'])
raw_data = {'random_seed' : [0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'precision': [0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'recall' : [0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'f1err' : [0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'accuracy' : [0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'delay' : [0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}
SVM_results = pd.DataFrame(raw_data, columns = ['random_seed', 'precision', 'recall', 'f1err', 'accuracy', 'delay'])
raw_data = {'random_seed' : [0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'precision': [0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'recall' : [0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'f1err' : [0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'accuracy' : [0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'delay' : [0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}
MAXENT_results = pd.DataFrame(raw_data, columns = ['random_seed', 'precision', 'recall', 'f1err', 'accuracy', 'delay'])
# load the library for train_test splitting
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler
#from nolearn.dbn import DBN 

for i in range(0,10):
    # split the train test features (histogram)
    # split train and test features with certain percentage
    X_train, X_test, y_train, y_test = train_test_split(X_train_main, y_train_main, test_size=0.20, random_state=randoms[i])
    #y_train = y_train[:,0]
    #y_test = y_test[:,0]
    scaling = MinMaxScaler(feature_range=(-1,1)).fit(X_train)
    X_train_scaled = scaling.transform(X_train)
    X_test_scaled = scaling.transform(X_test)
    X_test_scaled = np.asarray(X_test_scaled)    
    #KNN CLASSIFICATION:
    # Apply the KNN Classification
    # Create object
    model = neighbors.KNeighborsClassifier()
    start_secs = int(round(time.time()))
    # Train the model using the training sets
    model.fit(X_train_scaled, y_train)
    # Make predictions using the testing set
    y_pred = model.predict(X_test_scaled)
    finish_secs = int(round(time.time()))
    delay = finish_secs - start_secs;
    classifier_name = "KNN"
    cm = confusion_matrix(y_pred,y_test)
    precision, recall, F1err, accuracy = analyzeResults(cm,classifier_name)
    KNN_results['random_seed'][i]= randoms[i]
    KNN_results['precision'][i]= precision*100
    KNN_results['recall'][i]= recall*100
    KNN_results['f1err'][i]= F1err*100
    KNN_results['accuracy'][i]= accuracy*100
    KNN_results['delay'][i]= delay
    #NAIVE BAYES CLASSIFICATION:
    # Apply the NB Classification
    # Create object
    model = GaussianNB()
    start_secs = int(round(time.time()))
    # Train the model using the training sets
    model.fit(X_train_scaled, y_train)
    # Make predictions using the testing set
    y_pred = model.predict(X_test_scaled)
    finish_secs = int(round(time.time()))
    delay = finish_secs - start_secs;
    classifier_name = "NAIVE_BAYES"
    cm = confusion_matrix(y_pred,y_test)
    precision, recall, F1err, accuracy = analyzeResults(cm,classifier_name)
    NB_results['random_seed'][i]= randoms[i]
    NB_results['precision'][i]= precision*100
    NB_results['recall'][i]= recall*100
    NB_results['f1err'][i]= F1err*100
    NB_results['accuracy'][i]= accuracy*100
    NB_results['delay'][i]= delay
    #SVM CLASSIFICATION:
    # Apply the SVM Classification
    # Create object
    model = svm.SVC(kernel='linear', C=1)
    start_secs = int(round(time.time()))
    # Train the model using the training sets
    model.fit(X_train_scaled, y_train)
    # Make predictions using the testing set
    y_pred = model.predict(X_test_scaled)
    finish_secs = int(round(time.time()))
    delay = finish_secs - start_secs;
    classifier_name = "SVM"
    cm = confusion_matrix(y_pred,y_test)
    precision, recall, F1err, accuracy = analyzeResults(cm,classifier_name)
    SVM_results['random_seed'][i]= randoms[i]
    SVM_results['precision'][i]= precision*100
    SVM_results['recall'][i]= recall*100
    SVM_results['f1err'][i]= F1err*100
    SVM_results['accuracy'][i]= accuracy*100
    SVM_results['delay'][i]= delay
    
    #LOGISTIC REGRESSION (AKA MAXIMUM ENTROPY or MAXENT) CLASSIFICATION:
    # Apply the maxent Classification
    from sklearn import linear_model
    # Create object
    model = linear_model.LogisticRegression(C=1e5)
    start_secs = int(round(time.time()))
    # Train the model using the training sets
    model.fit(X_train_scaled, y_train)
    # Make predictions using the testing set
    y_pred = model.predict(X_test_scaled)
    finish_secs = int(round(time.time()))
    delay = finish_secs - start_secs;
    classifier_name = "LOGISTIC_REGRESSION"
    cm = confusion_matrix(y_pred,y_test)
    precision, recall, F1err, accuracy = analyzeResults(cm,classifier_name)
    MAXENT_results['random_seed'][i]= randoms[i]
    MAXENT_results['precision'][i]= precision*100
    MAXENT_results['recall'][i]= recall*100
    MAXENT_results['f1err'][i]= F1err*100
    MAXENT_results['accuracy'][i]= accuracy*100
    MAXENT_results['delay'][i]= delay

fname = 'Main_Results_KNN_'+split_size +'.csv'
KNN_results.to_csv(fname, index=False, header=False)
fname = 'Main_Results_NB_'+split_size +'.csv'
NB_results.to_csv(fname, index=False, header=False)
fname = 'Main_Results_SVM_'+split_size +'.csv'
SVM_results.to_csv(fname, index=False, header=False)
fname = 'Main_Results_MATENT_'+split_size +'.csv'
MAXENT_results.to_csv(fname, index=False, header=False)