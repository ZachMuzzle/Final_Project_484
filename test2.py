# used for manipulating directory paths
import os

# Scientific and vector computation for python
import matplotlib.pyplot as plt
import numpy as np

# Plotting library
from matplotlib import pyplot

# Optimization module in scipy
from scipy import optimize

# library written for this exercise providing additional functions for assignment submission, and others
import utils
from numpy import genfromtxt
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
#bank_data = np.loadtxt(open('bank-additional-full.csv','r'),dtype='str',delimiter=';',skiprows=1)
#print(bank_data.head())
#X,y = bank_data[:,0:20], bank_data[:,20]
def sigmoid(z):
    z = np.array(z)
    out = np.zeros(z.shape)
    out = 1/(1+np.exp(-z))
    return out

def cost_function(theta,X,y):
    m = y.size

    J = 0
    grad = np.zeros(theta.shape)

    temp = sigmoid(np.dot(X,theta))
    #print(temp)
    J = (1 / m) * np.sum((-y * np.log(temp)) + (-(1 - y) * np.log(1 - temp)))

    grad = 1 / m * np.dot(X.T, (temp - y))

    return J, grad

def predict(theta,X):
    m = X.shape[0]

    predict = np.zeros(m)

    predict = sigmoid(np.dot(X,theta)) >= 0.5

    return predict

def predict_test(theta_test,X_test):
    m = X_test.shape[0]

    predict = np.zeros(m)

    predict = sigmoid(np.dot(X_test,theta_test)) >= 0.5

    return predict
bank_train_org = pd.read_csv('bank-additional-full.csv')
bank_train = bank_train_org.copy()
columns = bank_train.columns.values[0].split(';')
columns = [column.replace('"', '') for column in columns]
bank_train = bank_train.values
bank_train = [items[0].split(';') for items in bank_train]
bank_train = pd.DataFrame(bank_train,columns = columns)

#This replaces "" around these cats with nothing so they can be read properly
bank_train['job'] = bank_train['job'].str.replace('"', '')
bank_train['marital'] = bank_train['marital'].str.replace('"', '')
bank_train['education'] = bank_train['education'].str.replace('"', '')
bank_train['default'] = bank_train['default'].str.replace('"', '')
bank_train['housing'] = bank_train['housing'].str.replace('"', '')
bank_train['loan'] = bank_train['loan'].str.replace('"', '')
bank_train['contact'] = bank_train['contact'].str.replace('"', '')
bank_train['month'] = bank_train['month'].str.replace('"', '')
bank_train['day_of_week'] = bank_train['day_of_week'].str.replace('"', '')
bank_train['poutcome'] = bank_train['poutcome'].str.replace('"', '')
bank_train['y'] = bank_train['y'].str.replace('"', '')
#print(bank_train['age'])
bank_train['age'] = bank_train['age'].astype('int64')
bank_train['duration'] = bank_train['duration'].astype('int64')
bank_train['campaign'] = bank_train['campaign'].astype('int64')
bank_train['pdays'] = bank_train['pdays'].astype('int64')
bank_train['previous'] = bank_train['previous'].astype('int64')
bank_train['emp.var.rate'] = bank_train['emp.var.rate'].astype('float') # negative
bank_train['emp.var.rate'] = bank_train['emp.var.rate'].abs()
bank_train['cons.price.idx'] = bank_train['cons.price.idx'].astype('float')
bank_train['cons.conf.idx'] = bank_train['cons.conf.idx'].astype('float') #negative
bank_train['cons.conf.idx'] = bank_train['cons.conf.idx'].abs()
bank_train['euribor3m'] = bank_train['euribor3m'].astype('float')
bank_train['nr.employed'] = bank_train['nr.employed'].astype('float')
print(bank_train['emp.var.rate'].value_counts())
#bank_train['job'] = bank_train['job'].astype('int64')

#Turn objects to ints
#print(bank_train['job'].head())
#job
bank_train['job'] = bank_train['job'].astype('category')
bank_train['job'] = bank_train['job'].cat.codes
bank_train['job'] = bank_train['job'].astype('int64')
#marital
bank_train['marital'] = bank_train['marital'].astype('category')
bank_train['marital'] = bank_train['marital'].cat.codes
bank_train['marital'] = bank_train['marital'].astype('int64')
#education
bank_train['education'] = bank_train['education'].astype('category')
bank_train['education'] = bank_train['education'].cat.codes
bank_train['education'] = bank_train['education'].astype('int64')
#Default
bank_train['default'] = bank_train['default'].astype('category')
bank_train['default'] = bank_train['default'].cat.codes
bank_train['default'] = bank_train['default'].astype('int64')
#Housing
bank_train['housing'] = bank_train['housing'].astype('category')
bank_train['housing'] = bank_train['housing'].cat.codes
bank_train['housing'] = bank_train['housing'].astype('int64')
#Loan
bank_train['loan'] = bank_train['loan'].astype('category')
bank_train['loan'] = bank_train['loan'].cat.codes
bank_train['loan'] = bank_train['loan'].astype('int64')
#Contact
bank_train['contact'] = bank_train['contact'].astype('category')
bank_train['contact'] = bank_train['contact'].cat.codes
bank_train['contact'] = bank_train['contact'].astype('int64')
#Month
bank_train['month'] = bank_train['month'].astype('category')
bank_train['month'] = bank_train['month'].cat.codes
bank_train['month'] = bank_train['month'].astype('int64')
#Day of week
bank_train['day_of_week'] = bank_train['day_of_week'].astype('category')
bank_train['day_of_week'] = bank_train['day_of_week'].cat.codes
bank_train['day_of_week'] = bank_train['day_of_week'].astype('int64')
#poutcome
bank_train['poutcome'] = bank_train['poutcome'].astype('category')
bank_train['poutcome'] = bank_train['poutcome'].cat.codes
bank_train['poutcome'] = bank_train['poutcome'].astype('int64')


#y
print(bank_train['job'].head())
bank_train['y'] = bank_train['y'].astype('category')
bank_train['y'] = bank_train['y'].cat.codes
bank_train['y'] = bank_train['y'].astype('int64')
print(bank_train['y'].head())
#print(bank_train['cons.conf.idx'])
print(bank_train.head())
#print(bank_train.dtypes)
bank_train['y'].value_counts().plot.bar()
plt.show()
###############################################
#Test data
bank_test_org = pd.read_csv("bank-additional.csv", na_values =['NA'])
bank_test = bank_test_org.copy()
bank_test = bank_test.values
bank_test = [items[0].split(';') for items in bank_test]
bank_test = pd.DataFrame(bank_test,columns=columns)

bank_test['job'] = bank_test['job'].str.replace('"', '')
bank_test['marital'] = bank_test['marital'].str.replace('"', '')
bank_test['education'] = bank_test['education'].str.replace('"', '')
bank_test['default'] = bank_test['default'].str.replace('"', '')
bank_test['housing'] = bank_test['housing'].str.replace('"', '')
bank_test['loan'] = bank_test['loan'].str.replace('"', '')
bank_test['contact'] = bank_test['contact'].str.replace('"', '')
bank_test['month'] = bank_test['month'].str.replace('"', '')
bank_test['day_of_week'] = bank_test['day_of_week'].str.replace('"', '')
bank_test['poutcome'] = bank_test['poutcome'].str.replace('"', '')
bank_test['y'] = bank_test['y'].str.replace('"', '')

bank_test['age'] = bank_test['age'].astype('int64')
bank_test['duration'] = bank_test['duration'].astype('int64')
bank_test['campaign'] = bank_test['campaign'].astype('int64')
bank_test['pdays'] = bank_test['pdays'].astype('int64')
bank_test['previous'] = bank_test['previous'].astype('int64')
bank_test['emp.var.rate'] = bank_test['emp.var.rate'].astype('float') # negative
bank_test['emp.var.rate'] = bank_test['emp.var.rate'].abs()
bank_test['cons.price.idx'] = bank_test['cons.price.idx'].astype('float')
bank_test['cons.conf.idx'] = bank_test['cons.conf.idx'].astype('float') #negative
bank_test['cons.conf.idx'] = bank_test['cons.conf.idx'].abs()
bank_test['euribor3m'] = bank_test['euribor3m'].astype('float')
bank_test['nr.employed'] = bank_test['nr.employed'].astype('float')
#############################################
bank_test['job'] = bank_test['job'].astype('category')
bank_test['job'] = bank_test['job'].cat.codes
bank_test['job'] = bank_test['job'].astype('int64')
#marital
bank_test['marital'] = bank_test['marital'].astype('category')
bank_test['marital'] = bank_test['marital'].cat.codes
bank_test['marital'] = bank_test['marital'].astype('int64')
#education
bank_test['education'] = bank_test['education'].astype('category')
bank_test['education'] = bank_test['education'].cat.codes
bank_test['education'] = bank_test['education'].astype('int64')
#Default
bank_test['default'] = bank_test['default'].astype('category')
bank_test['default'] = bank_test['default'].cat.codes
bank_test['default'] = bank_test['default'].astype('int64')
#Housing
bank_test['housing'] = bank_test['housing'].astype('category')
bank_test['housing'] = bank_test['housing'].cat.codes
bank_test['housing'] = bank_test['housing'].astype('int64')
#Loan
bank_test['loan'] = bank_test['loan'].astype('category')
bank_test['loan'] = bank_test['loan'].cat.codes
bank_test['loan'] = bank_test['loan'].astype('int64')
#Contact
bank_test['contact'] = bank_test['contact'].astype('category')
bank_test['contact'] = bank_test['contact'].cat.codes
bank_test['contact'] = bank_test['contact'].astype('int64')
#Month
bank_test['month'] = bank_test['month'].astype('category')
bank_test['month'] = bank_test['month'].cat.codes
bank_test['month'] = bank_test['month'].astype('int64')
#Day of week
bank_test['day_of_week'] = bank_test['day_of_week'].astype('category')
bank_test['day_of_week'] = bank_test['day_of_week'].cat.codes
bank_test['day_of_week'] = bank_test['day_of_week'].astype('int64')
#poutcome
bank_test['poutcome'] = bank_test['poutcome'].astype('category')
bank_test['poutcome'] = bank_test['poutcome'].cat.codes
bank_test['poutcome'] = bank_test['poutcome'].astype('int64')
#y
print(bank_test['job'].head())
bank_test['y'] = bank_test['y'].astype('category')
bank_test['y'] = bank_test['y'].cat.codes
bank_test['y'] = bank_test['y'].astype('int64')

########################################
##Train
X = bank_train[['age','job','marital','education','default','housing','loan','contact','month','day_of_week','duration','campaign','pdays','previous',
                'poutcome','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed']]

y = bank_train['y']
y = y.values

## Test
X_test = bank_test[['age','job','marital','education','default','housing','loan','contact','month','day_of_week','duration','campaign','pdays','previous',
                'poutcome','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed']]
y_test = bank_test['y']
y_test = y_test.values
print(X.dtypes)
#print(X.astype())
print(X_test.head())
#print(y.head())
m,n = X.shape
m_test, n_test = X_test.shape
X = np.concatenate([np.ones((m,1)),X], axis=1)
X_test = np.concatenate([np.ones((m_test,1)),X_test], axis=1)
print('Print y:\n ', y)
print('###############')
print(X)
initial_theta = np.zeros(n+1)
print(initial_theta)
initial_theta_test = np.zeros(n_test+1)
cost, grad = cost_function(initial_theta,X,y)
cost_test, grad_test = cost_function(initial_theta_test,X_test,y_test)
###########################################
########### optimize

options = {'maxiter': 400}

res = optimize.minimize(cost_function,
                        initial_theta,
                        (X, y),
                        jac=True,
                        method='TNC',
                        options=options)
res_test = optimize.minimize(cost_function,
                        initial_theta_test,
                        (X_test, y_test),
                        jac=True,
                        method='TNC',
                        options=options)

cost = res.fun

theta = res.x

cost_test = res_test.fun

theta_test = res_test.x

print('Cost at theta found by optimize.minimize: {:.3f}'.format(cost))
print('Expected cost (approx): 0.203\n');

print('theta:')
print('\t[{:.3f}, {:.3f}, {:.3f}]'.format(*theta))

predict = predict(theta,X)
print('Train Accuracy: {:.2f} %'.format(np.mean(predict == y) * 100))
cm = confusion_matrix(y,predict)
print('Confusion Matrix: \n', cm)

#print('Classfication Report: ', classification_report(y,predict))
predict_test = predict_test(theta_test,X_test)
print('Test Accuracy: {:.2f} %'.format(np.mean(predict_test == y_test) * 100))
cm2 = confusion_matrix(y_test,predict_test)
print(cm2)



