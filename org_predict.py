from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import numpy as np
import pathlib
import cv2
import os

org_dir = './org_img'
org_dir = pathlib.Path(org_dir)

class_names = np.array([item.name for item in org_dir.glob('*')])

# image
# image_path = './Predict/wild_boar_2917_detections.jpg'

# image folder
folder_path = './Predict/'

# path to model
model_path = './models/3_128x3_model_Dec16.h5'

# dimensions of images
img_width, img_height = 128, 128

# load the model we saved
model = load_model(model_path)

# load all images into a list
images = []

for img in os.listdir(folder_path):
    img = os.path.join(folder_path, img)
    img = cv2.imread(img)
    img = cv2.resize(img, (img_width, img_height))
    img = img.astype("float") / 255.0
    img = np.reshape(img, [1, img_width, img_height, 3])
    images.append(img)  # Could be commented out

# stack up images list to pass for prediction
images = np.vstack(images)

prediction = model.predict(images)
print(prediction)

classes = np.argmax(model.predict(images), axis=-1)
# classes = np.argmax(model.predict(images))
print(classes)

names = [class_names[i] for i in classes]
print(names)
