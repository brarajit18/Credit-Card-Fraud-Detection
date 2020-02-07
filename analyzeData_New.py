# -*- coding: utf-8 -*-

"""
Credit card fraud detection
This notebook will test different methods on skewed data. The idea is to compare if preprocessing techniques work better when there is an overwhelming majority class that can disrupt the efficiency of our predictive model.
You will also be able to see how to apply cross validation for hyperparameter tuning on different classification models. My intention is to create models using:
Logistic Regression
We also want to have a try at anomaly detection techniques, but I still have to investigate a bit on that, so any advise will be appreciated!
"""
"""
OVERALL SCENARIO:
Clearly the data is totally unbalanced!!
This is a clear example where using a typical accuracy score to evaluate our classification algorithm. For example, if we just used a majority class to assign values to all records, we will still be having a high accuracy, BUT WE WOULD BE CLASSIFYING ALL "1" INCORRECTLY!!
There are several ways to approach this classification problem taking into consideration this unbalance.
Collect more data? Nice strategy but not applicable in this case
Changing the performance metric:
Use the confusio nmatrix to calculate Precision, Recall
F1score (weighted average of precision recall)
Use Kappa - which is a classification accuracy normalized by the imbalance of the classes in the data
ROC curves - calculates sensitivity/specificity ratio.
Resampling the dataset
Essentially this is a method that will process the data to have an approximate 50-50 ratio.
One way to achieve this is by OVER-sampling, which is adding copies of the under-represented class (better when you have little data)
Another is UNDER-sampling, which deletes instances from the over-represented class (better when he have lot's of data)
"""
# import libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

"""
Approach
We are not going to perform feature engineering in first instance. The dataset has been downgraded in order to contain 30 features (28 anonamised + time + amount).
We will then compare what happens when using resampling and when not using it. We will test this approach using a simple logistic regression classifier.
We will evaluate the models by using some of the performance metrics mentioned above.
We will repeat the best resampling/not resampling method, by tuning the parameters in the logistic regression classifier.
We will finally perform classifications model using other classification algorithms.
"""
# loading the dataset
data = pd.read_csv("creditcard.csv")
data.head()

# find the target classes
count_classes = pd.value_counts(data['Class'], sort = True).sort_index()
count_classes.plot(kind = 'bar')
plt.title("Fraud class histogram")
plt.xlabel("Class")
plt.ylabel("Frequency")

# setting our input and target variables
# also apply the resampling
# Normalising the amount column. The amount column is not in line 
# with the anonimised features.
from sklearn.preprocessing import StandardScaler

data['normAmount'] = StandardScaler().fit_transform(data['Amount'].reshape(-1, 1))
data = data.drop(['Time','Amount'],axis=1)
data.head()


"""
2. Assigning X and Y. No resampling.
"""
# extracting the training and testing data
X = data.ix[:, data.columns != 'Class']
y = data.ix[:, data.columns == 'Class']

"""
3. Resampling.
As we mentioned earlier, there are several ways to resample skewed data. Apart from under and over sampling, there is a very popular approach called SMOTE (Synthetic Minority Over-Sampling Technique), which is a combination of oversampling and undersampling, but the oversampling approach is not by replicating minority class but constructing new minority class data instance via an algorithm.
In this notebook, we will use traditional UNDER-sampling. I will probably try to implement SMOTE in future versions of the code, but for now I will use traditional undersamplig.
The way we will under sample the dataset will be by creating a 50/50 ratio. This will be done by randomly selecting "x" amount of sample from the majority class, being "x" the total number of records with the minority class.
"""

# Number of data points in the minority class
number_records_fraud = len(data[data.Class == 1])
fraud_indices = np.array(data[data.Class == 1].index)

# Picking the indices of the normal classes
normal_indices = data[data.Class == 0].index

# Out of the indices we picked, randomly select "x" number (number_records_fraud)
random_normal_indices = np.random.choice(normal_indices, number_records_fraud, replace = False)
random_normal_indices = np.array(random_normal_indices)

# Appending the 2 indices
under_sample_indices = np.concatenate([fraud_indices,random_normal_indices])

# Under sample dataset
under_sample_data = data.iloc[under_sample_indices,:]

X_undersample = under_sample_data.ix[:, under_sample_data.columns != 'Class']
y_undersample = under_sample_data.ix[:, under_sample_data.columns == 'Class']

# Showing ratio
print("Percentage of normal transactions: ", len(under_sample_data[under_sample_data.Class == 0])/len(under_sample_data))
print("Percentage of fraud transactions: ", len(under_sample_data[under_sample_data.Class == 1])/len(under_sample_data))
print("Total number of transactions in resampled data: ", len(under_sample_data))

#Splitting data into train and test set. Cross validation will be used 
# when calculating accuracies.
from sklearn.cross_validation import train_test_split

# Whole dataset
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 0)

print("Number transactions train dataset: ", len(X_train))
print("Number transactions test dataset: ", len(X_test))
print("Total number of transactions: ", len(X_train)+len(X_test))

# Undersampled dataset
X_train_undersample, X_test_undersample, y_train_undersample, y_test_undersample = train_test_split(X_undersample
                                                                                                   ,y_undersample
                                                                                                   ,test_size = 0.3
                                                                                                   ,random_state = 0)
print("")
print("Number transactions train dataset: ", len(X_train_undersample))
print("Number transactions test dataset: ", len(X_test_undersample))
print("Total number of transactions: ", len(X_train_undersample)+len(X_test_undersample))

from sklearn.cross_validation import train_test_split

# Whole dataset
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 0)

print("Number transactions train dataset: ", len(X_train))
print("Number transactions test dataset: ", len(X_test))
print("Total number of transactions: ", len(X_train)+len(X_test))


# Undersampled dataset
X_train_undersample, X_test_undersample, y_train_undersample, y_test_undersample = train_test_split(X_undersample
                                                                                                   ,y_undersample
                                                                                                   ,test_size = 0.3
                                                                                                   ,random_state = 0)
print("")
print("Number transactions train dataset: ", len(X_train_undersample))
print("Number transactions test dataset: ", len(X_test_undersample))
print("Total number of transactions: ", len(X_train_undersample)+len(X_test_undersample))

"""
Logistic regression classifier - Undersampled data
We are very interested in the recall score, because that is the metric that will help us try to capture the most fraudulent transactions. If you think how Accuracy, Precision and Recall work for a confusion matrix, recall would be the most interesting:
Accuracy = (TP+TN)/total
Precision = TP/(TP+FP)
Recall = TP/(TP+FN)
As we know, due to the imbalacing of the data, many observations could be predicted as False Negatives, being, that we predict a normal transaction, but it is in fact a fraudulent one. Recall captures this.
Obviously, trying to increase recall, tends to come with a decrease of precision. However, in our case, if we predict that a transaction is fraudulent and turns out not to be, is not a massive problem compared to the opposite.
We could even apply a cost function when having FN and FP with different weights for each type of error, but let's leave that aside for now.
"""

    
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
from sklearn.metrics import mean_absolute_error

rotation = 1

#Apply MinMax scaler
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train1 = scaler.transform(X_train)
X_test1 = scaler.transform(X_test)


raw_data = {'classifier_name' : ['NB','KNN','SVM'], 'precision': [0.0, 0.0, 0.0], 'recall' : [0.0, 0.0, 0.0], 'f1err' : [0.0, 0.0, 0.0], 'accuracy' : [0.0, 0.0, 0.0], 'TP' : [0, 0, 0], 'TN' : [0, 0, 0], 'FP' : [0, 0, 0], 'FN' : [0, 0, 0],'MAE' : [0.0,0.0,0.0], 'RAE' : [0.0, 0.0, 0.0]}
Overall_results = pd.DataFrame(raw_data, columns = ['classifier_name', 'precision', 'recall', 'f1err', 'accuracy', 'TP', 'TN', 'FP', 'FN','MAE','RAE'])


#Naive Bayes
classifier = MultinomialNB()
classifier.fit(X_train1, y_train)
labels = classifier.predict(X_test1)
print("Naive Bayes Results:")
cm = confusion_matrix(labels,y_test)
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
Overall_results['MAE'][iiddx]=mean_absolute_error(labels,y_test)
         
# KNN CLassifier
classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(X_train1, y_train)
labels = classifier.predict(X_test1)
print("KNN Results:")
cm = confusion_matrix(labels,y_test)
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
Overall_results['MAE'][iiddx]=mean_absolute_error(labels,y_test)

print (Overall_results)
fname = 'Overall_results_'+str(rotation)+'.csv'
Overall_results.to_csv(fname, index=False, header=False)
               
# SVM CLassifier
classifier = LinearSVC()
classifier.fit(X_train1, y_train)
labels = classifier.predict(X_test1)
print("SVM Results:")
cm = confusion_matrix(labels,y_test)
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
Overall_results['MAE'][iiddx]=mean_absolute_error(labels,y_test)

print (Overall_results)
fname = 'Overall_results_'+str(rotation)+'.csv'
Overall_results.to_csv(fname, index=False, header=False)