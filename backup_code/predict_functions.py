# Functions and globals for loading and running the Serengeti models

from datetime import datetime
from pathlib import Path
import os
import imghdr
import numpy as np
import pandas as pd
from tensorboard.plugins.hparams.api_pb2 import DatasetType
from tensorflow.keras.models import Model, load_model
import cv2 as cv
import matplotlib.pyplot as plt


MODEL_PATH_NAME = "model/weights.13-0.84.hdf5"
DATA_PATH_LWF = Path("predict")


def get_test_images_from_folder(data_path):
    test_img_list = [str(Path(data_path)/file) for file in os.listdir(data_path) \
                     if os.path.isfile(Path(data_path)/file) and imghdr.what(Path(data_path)/file) in ["jpeg", "png"]]
    print(f"Found {len(test_img_list)} images in folder: {data_path}.")
    return test_img_list


def load_trained_model(test_img_list):
    print(f"Loading model: {MODEL_PATH_NAME}.")
    print(f"Running inference on {len(test_img_list)} images.")
    model = load_model(MODEL_PATH_NAME)
    model.callback_fns = []
    print(f"model loaded.")
    return model


def run_inference(learn):
    inference_start = datetime.now()
    print(f"Starting inference. time={inference_start}")
    # preds,y = learn.get_preds(ds_type=DatasetType.validate)
    preds = learn.predict(x)
    inference_stop = datetime.now()
    print(f"Inference complete. It took {inference_stop - inference_start}.")
    return preds


def plot_predictions(test_img_list, pred_dicts):

    markings_color = (0.667, 0.686, 0.694)
    content_color = (0.106, 0.565, 0.969)
    bg_color = (0.388, 0.416, 0.435)

    fig, axs = plt.subplots(len(test_img_list),
                            2,
                            figsize=(12, 4 * len(test_img_list)),
                            gridspec_kw={'width_ratios': [4, 1]},
                            constrained_layout=True,
                            squeeze=False)
    fig.suptitle('Top 5 predictions per image',
                 color=markings_color,
                 va="bottom",
                 ha="right",
                 x=0.95,
                 y=1)
    fig.set_facecolor(bg_color)

    for i in range(len(test_img_list)):
        data = pred_dicts[i]
        names = list(data.keys())[::-1]
        values = [round(v,4) for v in list(data.values())[::-1]]

        img = cv.cvtColor(cv.imread(test_img_list[i]), cv.COLOR_BGR2RGB)

        axs[i,0].imshow(img)
        axs[i,0].set_axis_off()

        axs[i,1].set_facecolor(bg_color)
        axs[i,1].barh(names, values, color=content_color)
        axs[i,1].set_yticklabels(names, minor=False)
        for j, v in enumerate(values):
            axs[i,1].text(v + 0.01, j, str(v), va='center', color=content_color)
        axs[i,1].tick_params(color=markings_color, labelcolor=markings_color)
        for spine in axs[i,1].spines.values():
            spine.set_edgecolor(markings_color)
        axs[i,1].spines["top"].set_visible(False)
        axs[i,1].spines["right"].set_visible(False)
        axs[i,1].spines["left"].set_visible(False)


def run_classification(images_from="lwf"):
    if images_from == "lwf":
        test_img_list = get_test_images_from_folder(data_path=DATA_PATH_LWF)
    else:
        raise Exception('Please choose: images_from="serengeti", images_from="fun_exmaples" or images_from="upload" ('
                        'only available on Google Colab)')
    learn = load_trained_model(test_img_list)
    preds = run_inference(learn)
    classes = learn.data.classes
    preds_df = pd.DataFrame(
            np.stack(preds),
            index=test_img_list,
            columns=classes,
        )
    pred_dicts = [dict(preds_df.loc[img].nlargest()) for img in test_img_list]
    plot_predictions(test_img_list, pred_dicts)
