# used for manipulating directory paths
import os
# Scientific and vector computation for python
import numpy as np
# Plotting library
import  matplotlib.pyplot as plt
# Optimization module in scipy
import seaborn as sns
from scipy import optimize
# library written for this exercise providing additional functions for assignment submission, and others
import utils
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')

#preprocess the data # didn't use
def categorize(data):
    data_copy = data.copy()
    le = preprocessing.LabelEncoder()

    data_copy['job'] = le.fit_transform(data_copy['job'])
    data_copy['marital'] = le.fit_transform(data_copy['marital'])
    data_copy['education'] = le.fit_transform(data_copy['education'])
    data_copy['default'] = le.fit_transform(data_copy['default'])
    data_copy['housing'] = le.fit_transform(data_copy['housing'])
    data_copy['month'] = le.fit_transform(data_copy['month'])
    data_copy['loan'] = le.fit_transform(data_copy['loan'])
    data_copy['contact'] = le.fit_transform(data_copy['contact'])
    data_copy['day_of_week'] = le.fit_transform(data_copy['day_of_week'])
    data_copy['poutcome'] = le.fit_transform(data_copy['poutcome'])
    data_copy['y'] = le.fit_transform(data_copy['y'])
    return data_copy

#Load our data
bank_train = pd.read_csv("bank-additional-full.csv", na_values =['NA'])
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

#print(bank_train.head())
#Do the same for our test data
bank_test = pd.read_csv("bank-additional.csv", na_values =['NA'])
bank_test = bank_test.values
bank_test = [items[0].split(';') for items in bank_test]
bank_test = pd.DataFrame(bank_test,columns=columns)

#This replaces "" around these cats with nothing so they can be read properly
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
#print(bank_test.head())

our_data = pd.concat([bank_train,bank_test])
# replace all of basic cat with just basic
our_data.replace(['basic.6y','basic.4y', 'basic.9y'], 'basic', inplace=True)
#Used to check if we have any null values
#print(our_data.isnull().sum())

#print(our_data.describe().T)

#Hold all our categorical data and loop through it.

cat_col = [n for n in our_data.columns if our_data[n].dtypes == 'object']
for col in cat_col:
    print(col, '\n\n')
    #print our data with how much for each varaible.
    print(our_data[col].value_counts())
    print("=============" * 2)

#check for subscription
no_sub = len(our_data[our_data['y'] == 'no'])
yes_sub = len(our_data[our_data['y'] == 'yes'])
percent_no_sub = (no_sub/len(our_data['y'])) * 100
percent_yes_sub = (yes_sub/len(our_data['y'])) * 100

print('% of no sub: ', percent_no_sub)
print('% of yes sub: ', percent_yes_sub)

#graph data of subscription or not
our_data['y'].value_counts().plot.bar()
plt.show()
#show yes or no output for each catergory
'''
for col in cat_col:
    pd.crosstab(our_data[col],our_data.y).plot(kind='bar',figsize=(15,10))
    plt.title(col)
    plt.show()
    '''
'''
plt.figure(figsize=(10,6))
sns.distplot(a=our_data['age'],kde=False)
plt.show()
'''
# Must accout for 999 is pdays as it means they were not contacted
#print(our_data['pdays'].value_counts) # Showing pdays count
#print(our_data['pdays'] == 3)
our_data['pdays_no_contact'] = (our_data['pdays'] == 999) * 1
contact = ({'cellular':0, 'telephone':1})
our_data['contact'] = our_data['contact'].map(contact)
print(our_data['contact'].value_counts())
#encode the categorical variables using get_dummies
our_data = pd.get_dummies(our_data, columns = ['job','marital','education','default','housing','loan','month','day_of_week','poutcome'],drop_first=True)
#print(our_data)
#print(our_data['pdays_no_contact'].value_counts())
#print(our_data['pdays'].value_counts())
#split our data
X = our_data.loc[:,our_data.columns != 'y']
y = our_data.loc[:,our_data.columns == 'y']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.10, random_state=0)


#length of our data for train and test
print('Length of x train: ', len(X_train), '\n Length of y train: ', len(y_train))
print('Length of x test: ', len(X_test), '\n Length of y test: ', len(y_test))

#Normailze our data

norm = StandardScaler()
X_train = norm.fit_transform(X_train)
X_test = norm.fit_transform(X_test)

#display model
#print(X_train['age'])

#test our data
lr = LogisticRegression()
lr.fit(X_train,y_train)
print('Our train accuracy: ', lr.score(X_train,y_train))
print('Our test accuracy: ',lr.score(X_test,y_test))
#Create a y prediction
y_pred = lr.predict(X_test)
#Get accuracy from built in function
accuracy = metrics.accuracy_score(y_test,y_pred)
#Get percent of accuracy
accuracy_percent = 100 * accuracy
print('Accuracy is: ', accuracy_percent)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,y_pred)
print(cm)
