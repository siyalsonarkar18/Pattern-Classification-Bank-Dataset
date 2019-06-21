#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 12:32:19 2018

@author: siyalsonarkar
"""

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import numpy as np
import pandas as pd
from pandas import Series,DataFrame
from sklearn import preprocessing 
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import f1_score
from sklearn.naive_bayes import GaussianNB
#from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from imblearn.over_sampling import SMOTE


# Importing the data
X = pd.read_csv('bank-additional.csv',sep=',',header='infer')
label_train = X.iloc[:,19]


# Replacing the missing data with 0
Xnew  = pd.DataFrame.replace(X.iloc[:,0:20],'unknown','0')

# Encoding the label (Labelling the data)
le = preprocessing.LabelEncoder()
for i in range(1,10):
    Xnew.iloc[:,i] = le.fit_transform(Xnew.iloc[:,i])

Xnew.iloc[:,13] = le.fit_transform(Xnew.iloc[:,13])
Xnew.iloc[:,19] = le.fit_transform(Xnew.iloc[:,19])

# chnaging dataframe to float64 type
X2 = np.zeros(shape=(4119,20),dtype=np.float64)
X1 = np.zeros(shape=(4119,19),dtype=np.float64)
for i in range(0,20):
    X2[:,i] = Xnew.iloc[:,i]
#X1=np.array(X2)
for i in range(0,19):
    X1[:,i] = Xnew.iloc[:,i]
X1 = np.array(X1)

# separating the class label 
y = X2[:,19]

# spliting the train and test data 
X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size = 0.25)


# filling the missing data
imp = preprocessing.Imputer(missing_values = 0, strategy = 'most_frequent')
Y=imp.fit_transform(X_train[:,1:7])
X_train[:,1:7] = Y

# oversampling
sm = SMOTE(random_state=42)
X_train, y_train = sm.fit_sample(X_train, y_train)

# computing one hot encoder 
hot = preprocessing.OneHotEncoder(n_values='auto', categorical_features = [1,2,3,4,5,6,7,8,9,13],sparse=False)
hot.fit(X_train)
X_train = hot.transform(X_train)
X_test = hot.transform(X_test)
        
# Normalize the data 
scaler = preprocessing.StandardScaler()
scaler.fit(X_train[:,46:55])
X_train[:,46:55] = scaler.transform(X_train[:,46:55])
X_test[:,46:55] = scaler.transform(X_test[:,46:55])

# feature selection and extraction
pca = PCA()
pca.fit(X_train)
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)

# cross-validation
maximum = -1000
Cs = np.logspace(-2,2,10)
gammas = np.logspace(-2,2,10)
save_avg_acc = np.empty([10,10])
std_dev = np.empty([10,10])
for i in range(0,10):
    gamma = gammas[i]
    for j in range(0,10):
        C = Cs[j]
    #for 5 fold CV
        skf = StratifiedKFold(n_splits=5,shuffle=True)
        all_acc=[]
        cnt = 1
        for train_index, dev_index in skf.split(X_train,y_train):
            X_cv_train, X_cv_dev = X_train[train_index], X_train[dev_index]
            Y_cv_train, Y_cv_dev = y_train[train_index], y_train[dev_index]
            clf = SVC(C=C, kernel='rbf',gamma = gamma,class_weight = 'balanced')
            clf.fit(X_cv_train, Y_cv_train)
            Y_pred = clf.predict(X_cv_dev)
            acc = accuracy_score(Y_cv_dev, Y_pred)
            all_acc.append(acc)
            cnt +=1
            
        save_avg_acc[i][j] =np.mean(all_acc)
        std_dev[i][j] = np.std(all_acc)
        
#choose best value of C and gamma

for i in range(0,10):
    for j in range(0,10):  
        if save_avg_acc[i][j]> maximum:
            maximum = save_avg_acc[i][j]
            minimum = std_dev[i][j]
            indx_1 = i
            indx_2 = j
        if save_avg_acc[i][j] == maximum:
            if std_dev[i][j] < minimum:
                minimum = std_dev[i][j]
                indx_1 = i
                indx_2 = j
print('Best accuracy  = ',maximum ,'\nBest gamma =', gammas[indx_1], '\nBest C =', Cs[indx_2], '\nStd_dev =', minimum)

model = SVC(C=Cs[indx_2],kernel = 'rbf', gamma =gammas[indx_1], class_weight = 'balanced')
model.fit(X_train,y_train)
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)
accuracy_train = accuracy_score(y_train,y_pred_train)
print('accuracy_train: ', accuracy_train)
accuracy_test = accuracy_score(y_test,y_pred_test)
print('accuracy_test: ', accuracy_test)
print(classification_report(y_test,y_pred_test))
print(classification_report(y_train,y_pred_train))



# SVM classifier 
clf = SVC(C=1,kernel = 'linear', gamma =1, class_weight = 'balanced')
clf.fit(X_train,y_train)
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
accuracy_train = accuracy_score(y_train,y_pred_train)
print('accuracy_train: ', accuracy_train)
accuracy_test = accuracy_score(y_test,y_pred_test)
print('accuracy_test: ', accuracy_test)
print(classification_report(y_test,y_pred_test))
print('roc_auc_score :',roc_auc_score(y_test,y_pred_test))


# Naive Bayes' classifier 
print('For Naive Bayes :')
gnb = GaussianNB()
gnb.fit(X_train,y_train)
y_pred_train = gnb.predict(X_train)
y_pred_test = gnb.predict(X_test)
accuracy_train = accuracy_score(y_train,y_pred_train)
print('accuracy_train: ', accuracy_train)
accuracy_test = accuracy_score(y_test,y_pred_test)
print('accuracy_test: ', accuracy_test)
print(classification_report(y_test,y_pred_test))
print('roc_auc_score :',roc_auc_score(y_test,y_pred_test))
#print('F1 score: ', f1_score(y_test,y_pred_test,average='weighted'))



# Random forest classifier 
print('For Random Forest :')
clf = RandomForestClassifier()
clf.fit(X_train,y_train)
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
accuracy_train = accuracy_score(y_train,y_pred_train)
print('accuracy_train: ', accuracy_train)
accuracy_test = accuracy_score(y_test,y_pred_test)
print('accuracy_test: ', accuracy_test)
print(classification_report(y_test,y_pred_test))
print('roc_auc_score :',roc_auc_score(y_test,y_pred_test))



# Perceptron 
print('For Perceptron :')
clf = Perceptron(class_weight='balanced')
clf.fit(X_train,y_train)
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
accuracy_train = accuracy_score(y_train,y_pred_train)
print('accuracy_train: ', accuracy_train)
accuracy_test = accuracy_score(y_test,y_pred_test)
print('accuracy_test: ', accuracy_test)
print(classification_report(y_test,y_pred_test))
print('roc_auc_score :',roc_auc_score(y_test,y_pred_test))

# ROC curve
print('roc_auc_score :',roc_auc_score(y_test,y_pred_test))
fpr, tpr, threshold = roc_curve(y_test,y_pred_test)
r_a = auc(fpr,tpr)
roc_auc = roc_auc_score(y_test,y_pred_test)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

