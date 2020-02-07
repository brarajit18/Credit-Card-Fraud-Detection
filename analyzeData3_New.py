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

rotation = 1

#Apply MinMax scaler
scaler = MinMaxScaler()
scaler.fit(X_train_undersample.values)
X_train1 = X_train
y_train1 = y_train

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
    X_train, X_test, y_train, y_test = train_test_split(X_train1, y_train1, test_size=0.20, random_state=randoms[i])
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



