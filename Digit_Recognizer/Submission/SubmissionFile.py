import pandas as pd
import numpy as np

data = pd.read_csv('c:/users/wybek/documents/school/kaggle/Digit_Recognizer/Submission/Model_1.csv', header = 0, index_col = 0)
data = data.to_numpy()
results =[]
indices =[]
start = 1
for case in data:
    results.append(np.argmax(case))
    indices.append(start)
    start +=1

Submission = pd.DataFrame({'ImageId':indices,'Label': results})
Submission.to_csv('c:/users/wybek/documents/school/kaggle/Digit_Recognizer/Submission/Submission_Final.csv', index=False)