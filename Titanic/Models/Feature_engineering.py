import pandas as pd
import numpy as np
pd.set_option('display.max_columns', 500)

# Function for getting the most frequent occurence in a list.
def most_common(lst):
    return max(set(lst), key=lst.count)

data = pd.read_csv('C:/Users/wybek/Documents/school/Kaggle/Titanic/Data/train.csv')
#data = pd.read_csv('C:/Users/wybek/Documents/school/Kaggle/Titanic/Data/test.csv')

# Checking number of nans in each column.
for col in data.columns:
    print(col + " has the following number of nans: " + str(data[col].isna().sum()))

# Checking balance of labels. Labels are balanced.
Y = data['Survived']
print(str(sum(Y)) + " Survived our of " + str(len(Y)))

# Dropping three features I don't want to work with.
data = data.drop(labels=['Ticket', 'Cabin', 'Name'], axis=1)

# Change male and female to 1 and 0.
data['Sex'].replace(['female','male'], [0,1],inplace=True)

# If age is nan replace with mean age.
data['Age'].fillna((data['Age'].mean()), inplace=True)

# Replace nans in embarked with most common place of embarkment.
data['Embarked'].fillna(most_common(data['Embarked'].tolist()), inplace=True)

# Replace nan Fare with mean
data['Fare'].fillna(data['Fare'].mean(), inplace=True)

# Get set of all embarkment location
embarked_set = set(data['Embarked'].tolist())

# Replace char with number
data['Embarked'].replace(embarked_set, list(range(0, len(embarked_set))),inplace=True)

data.to_csv('C:/Users/wybek/Documents/school/Kaggle/Titanic/Data/test_processed.csv', index=False)

#https://stackoverflow.com/questions/38108832/passing-categorical-data-to-sklearn-decision-tree

