import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.constraints import max_norm



np.random.seed(20)

#loading in data
train = pd.read_csv('c:/Users/wybek/Documents/school/Kaggle/Aerial_Cactus_Identification/Data/train.csv')
train_dir = 'c:/Users/wybek/Documents/school/Kaggle/Aerial_Cactus_Identification/Data/train'
test_dir = 'c:/Users/wybek/Documents/school/Kaggle/Aerial_Cactus_Identification/Data/test'
train.has_cactus=train.has_cactus.astype(str)

#data augmentation and rescaling
datagen_train = ImageDataGenerator(
        rescale=1./255,
        rotation_range=190,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True)  # randomly flip images

datagen_val = ImageDataGenerator(rescale=1./255)


#setting up train and test generator from dataframe
train_generator = datagen_val.flow_from_dataframe(dataframe=train[:15000],directory=train_dir,x_col='id',
                                            y_col='has_cactus',class_mode='binary',batch_size=32,
                                            target_size=(32,32))

validation_generator = datagen_val.flow_from_dataframe(dataframe=train[15000:],directory=train_dir,x_col='id',
                                                y_col='has_cactus',class_mode='binary',batch_size=32,
                                                target_size=(32,32))


#Definining the model
model = Sequential()
model.add(Conv2D(32,(3,3),activation='relu',input_shape=(32,32,3), kernel_initializer='normal', kernel_constraint=max_norm(2)))
model.add(MaxPool2D((2,2)))
model.add(Conv2D(64,(3,3),activation='relu', kernel_initializer='normal', kernel_constraint=max_norm(2)))
model.add(MaxPool2D((2,2)))
model.add(Conv2D(128,(3,3),activation='relu', kernel_initializer='normal', kernel_constraint=max_norm(2)))
model.add(MaxPool2D((2,2)))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dense(128, activation='relu'))

model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer = 'Adam', loss = "binary_crossentropy", metrics=["accuracy"])

#adaptive learning rate
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',
                                            patience=3,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)

epochs = 100
history = model.fit(train_generator,steps_per_epoch=15000/32,epochs=epochs,validation_data=validation_generator, validation_steps=50, callbacks=[learning_rate_reduction], verbose=2)

# save the trained model
model.save('C:/Users/wybek/Documents/school/Kaggle/Aerial_Cactus_Identification/saved_models/Cactus_1.h5')
