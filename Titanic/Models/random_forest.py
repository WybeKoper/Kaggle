from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import pandas as pd
import os
print(os.getcwd())

# Load and finalize training data
train_data = pd.read_csv('C:/Users/wybek/Documents/school/Kaggle/Titanic/Data/train_processed.csv')
test_data = pd.read_csv('C:/Users/wybek/Documents/school/Kaggle/Titanic/Data/test_processed.csv')

Y = train_data['Survived']
train_data = train_data.drop(labels=['PassengerId', 'Survived'], axis=1)

# Training model and gettign out of bag estimate 98.2%
#model = RandomForestClassifier(n_estimators=1000, oob_score=True)
model = AdaBoostClassifier(n_estimators=500)
model.fit(train_data, Y)
print(model.score(train_data, Y))

# Create predictions
test_data = test_data.set_index('PassengerId')
predictions = model.predict(test_data)


submission = pd.DataFrame({'PassengerId': list(test_data.index), 'Survived': predictions})
submission.to_csv('C:/Users/wybek/Documents/school/Kaggle/Titanic/Submissions/Adaboost.csv', index=False)


