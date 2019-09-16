# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 16:14:12 2019

@author: Owner
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Import dataset
dataset = pd.read_csv('PS_20174392719_1491204439457_log.csv')
dataset.head()

#Check for any null values
dataset.isnull().any()

#Bar graph for transactions distribution
var = dataset.groupby('type').amount.sum()
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
var.plot(kind='bar')
ax1.set_title("Total amount per transaction type")
ax1.set_xlabel('Type of Transaction')
ax1.set_ylabel('Amount')

#Which transaction types have fraudulant behaviour
dataset.loc[dataset.isFraud==1].type.unique()

#finding correlation between variables
sns.heatmap(dataset.corr(),cmap='RdBu');

'''We notice a strong correlation between:
    1. OldBalanceOrg and NewBalanceOrg
    2. OldBalanceDest and NewBalanceDest'''

#Looking at the numbers: Data Analysis
dataset.shape
fraud = dataset.loc[dataset.isFraud == 1]
nonfraud = dataset.loc[dataset.isFraud == 0]
FlaggedFraud = dataset.loc[dataset.isFlaggedFraud == 1]
fraudcount = fraud.isFraud.count()
nonfraudcount = nonfraud.isFraud.count()
fraudDetected = dataset.isFlaggedFraud.sum()
total = 6362620
print('Total number of Transactions = '+ str(6362620))
print('Total number of fraud transactions = ' + str(fraudcount) )
print('Total number of Non-Faud Transactions = '+ str(nonfraudcount))
print('Ratio of Fraud transactions : Total Transactions = 1: ', format(int(total//fraudcount)))
print('Number of fraud transactions detected = ', format(int(fraudDetected)))
print('Number of Fraud transactions undetected = ', int(fraudcount-fraudDetected))
print('Amount of money lost due to fraud = $', format(int(fraud.amount.sum())))
print('Amount of money that could have been saved if we had a better model = ', format(int(fraud.amount.sum()-FlaggedFraud.amount.sum())))

#% of fraud detected
piedata = fraud.groupby(['isFlaggedFraud']).sum()
f, axes = plt.subplots(1,1, figsize=(6,6))
axes.set_title("% of fraud transaction detected")
piedata.plot(kind='pie',y='isFraud',ax=axes, fontsize=14,shadow=False,autopct='%1.1f%%');
axes.set_ylabel('');
plt.legend(loc='upper left',labels=['Not Detected','Detected'])
plt.show()

''' Thus only 0.2% of the fraud was actually detected'''

#Bivariate Analysis
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(fraud['step'],fraud['oldbalanceOrg'])
ax.scatter(fraud['step'],fraud['newbalanceOrig'])
plt.show()
'''We can see the huge drop in sender's balance due to fraud'''

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax1.scatter(fraud['step'],fraud['oldbalanceDest'])
ax1.scatter(fraud['step'],fraud['newbalanceDest'])
plt.show()
'''we see that there is no change in Receiver's old and new balances since the fraud sent money to third party'''

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(fraud['step'],fraud['oldbalanceOrg'])
ax.scatter(fraud['step'],fraud['newbalanceDest'])
plt.show()


#Data Cleaning
dataset_model= pd.read_csv('PS_20174392719_1491204439457_log.csv')
data_model = dataset_model.replace(to_replace={'PAYMENT':1,'TRANSFER':2,'CASH_OUT':3,'CASH_IN':4,'DEBIT':5,'No':0,'Yes':1})
data_model.drop(['nameOrig','nameDest','isFlaggedFraud'],axis=1,inplace=True)
data_model.head

#Data Splitting
X = data_model.drop(['isFraud'], axis =1 )
y = data_model[['isFraud']]
from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state= 42)


#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
model = rfc.fit(train_X,train_y.values.ravel())
predictions= model.predict(test_X)
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(test_y,predictions))
print(confusion_matrix(test_y, predictions))
from sklearn.metrics import accuracy_score
accuracy_score(test_y,predictions)
print("Random Forest Classifier Accuracy score is: ", accuracy_score(test_y,predictions))
''' Random Forest Classifier Accuracy score is:  0.9997155259940087'''

#Logistic Regression
from sklearn import linear_model
logitic = linear_model.LogisticRegression()
model = logitic.fit(train_X,train_y)
predictions = model.predict(test_X)
print(classification_report(test_y,predictions))
print(confusion_matrix(test_y, predictions))
accuracy=accuracy_score(test_y,predictions)
print("Logistic Regression Accuracy score is ", accuracy)
'''Logistic Regression Accuracy score is  0.9988259553454395'''

#Naive Bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
model = gnb.fit(train_X,train_y)
predictions = model.predict(test_X)
print(classification_report(test_y,predictions))
print(confusion_matrix(test_y, predictions))
accur= accuracy_score(test_y,predictions)
print("Naive Bayes Regression Accuracy score is ", accur)
'''Naive Bayes Regression Accuracy score is  0.9930146700573035'''






