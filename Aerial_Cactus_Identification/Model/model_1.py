import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image



import warnings
warnings.simplefilter(action='ignore', category = FutureWarning)



np.random.seed(20)

trainLabels = pd.read_csv('c:/Users/wybek/Documents/school/Kaggle/Aerial_Cactus_Identification/Data/train.csv')
train_dir = "'c:/Users/wybek/Documents/school/Kaggle/Aerial_Cactus_Identification/Data/train'"
test_dir = "'c:/Users/wybek/Documents/school/Kaggle/Aerial_Cactus_Identification/Data/test'"

#adjust imagdatagenerator
datagen = ImageDataGenerator(rescale=1./255)
batch_size = 150

train_generator = datagen.flow_from_dataframe(dataframe=trainLabels[:15001],directory=train_dir,x_col='id',
                                            y_col='has_cactus',class_mode='binary',batch_size=batch_size,
                                            target_size=(150,150))


validation_generator = datagen.flow_from_dataframe(dataframe=trainLabels[15000:],directory=train_dir,x_col='id',
                                                y_col='has_cactus',class_mode='binary',batch_size=50,
                                                target_size=(150,150))

#Definining the model
model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',
                 activation ='relu', input_shape = (150,150,3)))
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',
                 activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))

optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer = optimizer , loss = "binary_crossentropy", metrics=["accuracy"])

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                            patience=3,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)

datagen2 = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images


datagen2.fit(train_generator)

epochs=1
history = model.fit_generator(datagen2.flow(train_generator, batch_size=batch_size),
                              epochs = epochs, validation_data = validation_generator,
                              verbose = 0, steps_per_epoch =100
                              , callbacks=[learning_rate_reduction])

# save the trained model
model.save('c:/Users/wybek/Documents/school/Kaggle/Aerial_Cactus_Identifcation/Model/model_1.h5')