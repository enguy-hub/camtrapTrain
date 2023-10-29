from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import numpy as np
import pathlib
import cv2
import os

org_dir = '../org_img'
org_dir = pathlib.Path(org_dir)

class_names = np.array([item.name for item in org_dir.glob('*')])

# image
image_path = 'C:/Hien/Garden/LWF/CTC_Species/Predict/Capreolus_capreolus/gk3_2018-06-16_capreolus_capreolus_1954_detections.jpg'

# image folder
folder_path = 'C:/Hien/Garden/LWF/CTC_Species/Predict/'

# path to model
model_path = './models/5_64x3_model_Dec14.h5'

# dimensions of images
img_width, img_height = 64, 64

# load the model we saved
model = load_model(model_path)

img = cv2.imread(image_path,)
img = cv2.resize(img, (img_width, img_height))
img = img.astype("float") / 255.0
img = np.reshape(img, [1, img_width, img_height, 3])

classes = np.argmax(model.predict(img), axis=-1)

print(classes)

names = [class_names[i] for i in classes]

print(names)
