import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools



def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

model = tf.keras.models.load_model('c:/Users/wybek/Documents/school/Kaggle/Digit_Recognizer/Model/model_2.h5')


train = pd.read_csv('c:/Users/wybek/Documents/school/Kaggle/Digit_Recognizer/Data/train.csv')
test = pd.read_csv('c:/Users/wybek/Documents/school/Kaggle/Digit_Recognizer/Data/test.csv')

y_train = train['label']
x_train = train.drop(labels=['label'], axis=1)
del train

#Grayscale normalization
x_train = x_train / 255.0
test = test / 255.0

#Reshape data
x_train = x_train.to_numpy().reshape(-1, 28, 28, 1)
test = test.to_numpy().reshape(-1, 28, 28, 1)

#One hot encoding
y_train = to_categorical(y_train)



x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.1, random_state=3)


Y_pred = model.predict(x_val)

Y_pred_classes = np.argmax(Y_pred,axis = 1)

Y_true = np.argmax(y_val,axis = 1)

confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)

plot_confusion_matrix(confusion_mtx, classes=range(10))

# predict results
results = model.predict(test)

# select the indix with the maximum probability
results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("c:/Users/wybek/Documents/school/Kaggle/Digit_Recognizer/submission/model2submission.csv",index=False)
