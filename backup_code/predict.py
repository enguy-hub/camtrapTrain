# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import argparse


# For Avoiding GPU Errors
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True,
                help="path to out input directory of images")
ap.add_argument("-m", "--model", required=True,
                help="path to pre-trained model")
args = vars(ap.parse_args())


batch_size = 64
IMAGE_SHAPE = (64, 64, 3)


def load_data():
    # org_dir = './org_img'
    # org_dir = pathlib.Path(org_dir)

    pred_dir = '../Predict'
    pred_dir = pathlib.Path(pred_dir)

    class_names = np.array([item.name for item in pred_dir.glob('*')])

    # 20% validation set 80% training set
    image_generator = ImageDataGenerator(rescale=1 / 255)

    # make the training dataset generator
    predict_data_gen = image_generator.flow_from_directory(directory=str(pred_dir),
                                                           batch_size=batch_size,
                                                           classes=list(class_names),
                                                           target_size=(IMAGE_SHAPE[0], IMAGE_SHAPE[1]),
                                                           shuffle=True)

    return predict_data_gen, class_names


if __name__ == "__main__":

    # load the data generators
    predict_generator, class_names = load_data()

    print("[INFO] loading pre-trained network...")
    model = load_model(args["model"])

    # validation_steps_per_epoch = np.ceil(predict_generator.samples / batch_size)
    # print the validation loss & accuracy
    evaluation = model.evaluate_generator(predict_generator, verbose=1)  # steps=validation_steps_per_epoch
    print("Val loss:", evaluation[0])
    print("Val Accuracy:", evaluation[1])

    # get a random batch of images
    image_batch, label_batch = next(iter(predict_generator))
    # turn the original labels into human-readable text
    label_batch = [class_names[np.argmax(label_batch[i])] for i in range(batch_size)]
    # predict the images on the model
    predicted_class_names = model.predict(image_batch)
    predicted_ids = [np.argmax(predicted_class_names[i]) for i in range(batch_size)]
    # turn the predicted vectors to human readable labels
    predicted_class_names = np.array([class_names[id] for id in predicted_ids])

    # some nice plotting
    plt.figure(figsize=(10, 9))
    for n in range(20):
        plt.subplot(5, 4, n + 1)
        plt.subplots_adjust(hspace=0.3)
        plt.imshow(image_batch[n])
        if predicted_class_names[n] == label_batch[n]:
            color = "blue"
            title = predicted_class_names[n].title()
        else:
            color = "red"
            title = f"{predicted_class_names[n].title()}, correct:{label_batch[n]}"
        plt.title(title, color=color)
        plt.axis('off')
    _ = plt.suptitle("Model predictions (blue: correct, red: incorrect)")
    plt.show()
