# -*- coding: utf-8 -*-
##PROPOSED CLASSIFICATION MODEL
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

rotation = 1

#Apply MinMax scaler
scaler = MinMaxScaler()
scaler.fit(X_train_undersample.values)
X_train1 = scaler.transform(X_train_undersample.values)
X_test1  = scaler.transform(X_test_undersample.values)
y_train1 = y_train_undersample.values.ravel()
y_test1  = y_test_undersample.values.ravel()

raw_data = {'classifier_name' : ['NB','KNN','SVM'], 'precision': [0.0, 0.0, 0.0], 'recall' : [0.0, 0.0, 0.0], 'f1err' : [0.0, 0.0, 0.0], 'accuracy' : [0.0, 0.0, 0.0], 'TP' : [0, 0, 0], 'TN' : [0, 0, 0], 'FP' : [0, 0, 0], 'FN' : [0, 0, 0]}
Overall_results = pd.DataFrame(raw_data, columns = ['classifier_name', 'precision', 'recall', 'f1err', 'accuracy', 'TP', 'TN', 'FP', 'FN'])


#Naive Bayes
classifier = MultinomialNB()
classifier.fit(X_train1, y_train1)
labels = classifier.predict(X_test1)
print("Naive Bayes Results:")
cm = confusion_matrix(labels,y_test1)
TN = cm[0,0]
FP = cm[0,1]
FN = cm[1,0]
TP = cm[1,1]
accuracy = float(TP+TN)/(TP+FP+FN+TN)
precision = float(TP) / (TP+FP)
recall = float(TP) / (TP+FN)
f1err = 2 * (float (precision * recall) / (precision + recall))
#print(cm)
#print(str(accuracy*100) + "%")
iiddx = 0
Overall_results['precision'][iiddx]=precision
Overall_results['recall'][iiddx]=recall
Overall_results['f1err'][iiddx]=f1err
Overall_results['accuracy'][iiddx]=accuracy
Overall_results['TP'][iiddx]=TP
Overall_results['TN'][iiddx]=TN
Overall_results['FP'][iiddx]=FP
Overall_results['FN'][iiddx]=FN
         
# KNN CLassifier
classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(X_train1, y_train1)
labels = classifier.predict(X_test1)
print("KNN Results:")
cm = confusion_matrix(labels,y_test1)
TN = cm[0,0]
FP = cm[0,1]
FN = cm[1,0]
TP = cm[1,1]
accuracy = float(TP+TN)/(TP+FP+FN+TN)
precision = float(TP) / (TP+FP)
recall = float(TP) / (TP+FN)
f1err = 2 * (float (precision * recall) / (precision + recall))
#print(cm)
#print(str(accuracy*100) + "%")
iiddx = 1
Overall_results['precision'][iiddx]=precision
Overall_results['recall'][iiddx]=recall
Overall_results['f1err'][iiddx]=f1err
Overall_results['accuracy'][iiddx]=accuracy
Overall_results['TP'][iiddx]=TP
Overall_results['TN'][iiddx]=TN
Overall_results['FP'][iiddx]=FP
Overall_results['FN'][iiddx]=FN

print (Overall_results)
fname = 'Overall_results_'+str(rotation)+'.csv'
Overall_results.to_csv(fname, index=False, header=False)
               
# SVM CLassifier
classifier = LinearSVC()
classifier.fit(X_train1, y_train1)
labels = classifier.predict(X_test1)
print("SVM Results:")
cm = confusion_matrix(labels,y_test1)
TN = cm[0,0]
FP = cm[0,1]
FN = cm[1,0]
TP = cm[1,1]
accuracy = float(TP+TN)/(TP+FP+FN+TN)
precision = float(TP) / (TP+FP)
recall = float(TP) / (TP+FN)
f1err = 2 * (float (precision * recall) / (precision + recall))
#print(cm)
#print(str(accuracy*100) + "%")
iiddx = 2
Overall_results['precision'][iiddx]=precision
Overall_results['recall'][iiddx]=recall
Overall_results['f1err'][iiddx]=f1err
Overall_results['accuracy'][iiddx]=accuracy
Overall_results['TP'][iiddx]=TP
Overall_results['TN'][iiddx]=TN
Overall_results['FP'][iiddx]=FP
Overall_results['FN'][iiddx]=FN

print (Overall_results)
fname = 'Overall_results_'+str(rotation)+'.csv'
Overall_results.to_csv(fname, index=False, header=False)
