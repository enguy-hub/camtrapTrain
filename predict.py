# Functions and globals for loading and running the Serengeti models
from tensorflow.keras.models import load_model
from datetime import datetime
from pathlib import Path
import os
import cv2
import imghdr
import pandas as pd
import numpy as np


def get_convert_predict_images(data_path):

    img_list = [str(Path(data_path) / file) for file in os.listdir(data_path)
                if
                os.path.isfile(Path(data_path) / file) and imghdr.what(Path(data_path) / file) in ["jpeg", "png"]]
    print(f"Found {len(img_list)} images in folder: {data_path}.")

    # load all images into a list
    images = []
    for img in os.listdir(data_path):
        img = os.path.join(data_path, img)
        img = cv2.imread(img)
        img = cv2.resize(img, (128, 128))
        img = img.astype("float") / 255.0
        img = np.reshape(img, [1, 128, 128, 3])
        images.append(img)  # Could be commented out

    # stack up images list to pass for prediction
    images = np.vstack(images)  # use this in run_classification

    return img_list, images


def load_pretrained_model(model_path, img_list):

    print(f"Loading model: {model_path}.")
    print(f"Running inference on {len(img_list)} images.")

    model = load_model(model_path)
    model.summary()

    print(f"Pretrained-model loaded !!")

    return model


def run_inference(pretrained_model, images, img_list):

    inference_start = datetime.now()
    print(f"Running inference on {len(img_list)} images.")
    print(f"Starting inference. time={inference_start}")

    preds = pretrained_model.predict(images)

    inference_stop = datetime.now()
    print(f"Inference complete. It took {inference_stop - inference_start}.")

    return preds


def run_classification():

    # Define model path
    MODEL_PATH = "./Models/3_128x3_model_Dec16.h5"

    # Ask user to input predict and model directories paths
    INPUT_PREDICT_DIR = str(input("Enter the folder path where the images you want to predict are stored: "))

    # Get the return parameters from above function
    IMG_LIST, images = get_convert_predict_images(INPUT_PREDICT_DIR)

    # Define class names
    class_names = ["Capreolus_capreolus", "Human", "Sus_scrofa"]

    # Load important parameters for preds_df
    MODEL = load_pretrained_model(MODEL_PATH, IMG_LIST)
    PREDS = run_inference(MODEL, images, IMG_LIST)
    CLASSES = class_names

    preds_df = pd.DataFrame(
            np.stack(PREDS),
            index=IMG_LIST,
            columns=CLASSES,)

    print(preds_df)

    # Save results into json file
    preds_df.to_json(r'./prediction.json', orient='index', indent=4)


if __name__ == '__main__':

    run_classification()
