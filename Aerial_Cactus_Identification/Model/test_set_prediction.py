import numpy as np
import pandas as pd

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


datagen_test = ImageDataGenerator(rescale=1./255)
test_generator = datagen_test.flow_from_directory(
    directory='C:/Users/wybek/Documents/school/Kaggle/Aerial_Cactus_Identification/Data/test',
    target_size=(32, 32),
    color_mode="rgb",
    batch_size=32,
    class_mode=None,
    shuffle=False)




model = load_model('C:/Users/wybek/Documents/school/Kaggle/Aerial_Cactus_Identification/saved_models/Cactus_1.h5')

predicted = model.predict_generator(test_generator, steps=4000/32)
rounded_predictions = predicted.round()


preds = [item for rounded_predictions in rounded_predictions for item in rounded_predictions]
print(preds)
filenames = test_generator.filenames
cropped_filenames = list()
print(type(filenames))

for filename in filenames:
    cropped_filenames = filename[11:]

results=pd.DataFrame({"id":cropped_filenames, "has_cactus": preds})

print(results)

results.to_csv('C:/Users/wybek/Documents/school/Kaggle/Aerial_Cactus_Identification/Submission/version_1.csv')