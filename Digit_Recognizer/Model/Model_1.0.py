import numpy as np
import pandas as pd
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical


dataPath = 'c:/Users/wybek/Documents/school/Kaggle/Digit_Recognizer/Data/train.csv'
testPath = 'c:/Users/wybek/Documents/school/Kaggle/Digit_Recognizer/Data/test.csv'
data = pd.read_csv(dataPath)
test = pd.read_csv(testPath)

x_train = data.drop(columns=['label']).to_numpy()
y_train = data['label'].to_numpy()
y_train = to_categorical(y_train)
x_test = test.to_numpy()

# x_train = x_train / 255.0
# x_test = x_test / 255.0


model = Sequential()
model.add(Dense(100, input_dim=784, activation='softmax'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=10, batch_size=32)
classes = model.predict(x_test, batch_size=128)

submission = pd.DataFrame(classes)
submission.to_csv('c:/Users/wybek/Documents/school/Kaggle/Digit_Recognizer/Submission/Model_1.csv')

model.save('c:/Users/wybek/Documents/school/Kaggle/Digit_Recognizer/Model/model_1.h5')
