# Importing necessary libraries and python files
import os
import random
import shutil
import config


# Loop over the original images folder and split the images
def data_splitting():
    # Use class names for folder split
    for class_split in config.class_names:

        # Grab all image paths in the current split
        print("[INFO] Processing '{} class' split ...".format(class_split))

        # Create a path to 'original' folder and all sub-folders
        org_path = os.path.sep.join([config.original_data_dir, class_split])
        org_images_path = os.listdir(org_path)

        # Shuffle them up randomly
        random.seed(42)
        random.shuffle(org_images_path)

        # List all images in 'original' dir / [i for i in org_images_path]
        all_images = [i for i in org_images_path]
        print("Total images of this class: ", len(all_images))

        # Contribute 80% of total amount of images for training
        train_images = random.sample(all_images, int(len(all_images)*0.85))
        print("# Train images: ", len(train_images))

        # Contribute 20% of total amount of images for validating
        test_images = random.sample(all_images, int(len(all_images)*0.12))
        print("# Validate images: ", len(test_images))

        # # Contribute 2% of total amount of images for predicting
        predict_images = random.sample(all_images, int(len(all_images) * 0.03))
        print("# Predict images: ", len(predict_images))

        # Construct the paths to the output folders for classes inside train & validation directories
        train_dirPath = os.path.sep.join([config.train_dir, class_split])
        test_dirPath = os.path.sep.join([config.test_dir, class_split])
        # predict_dirPath = os.path.sep.join([config.predict_dir, class_split])
        predict_dirPath = os.path.join(config.predict_dir)

        # If the output directory does not exist, create it
        os.makedirs(train_dirPath) if not os.path.exists(train_dirPath) else None
        os.makedirs(test_dirPath) if not os.path.exists(test_dirPath) else None
        os.makedirs(predict_dirPath) if not os.path.exists(predict_dirPath) else None

        # Make copies of train, val, and predict images into their data folders
        for cp_train_images in train_images:
            src = os.path.join(org_path, cp_train_images)
            dst = os.path.join(train_dirPath, cp_train_images)
            shutil.copyfile(src, dst)
        for cp_test_images in test_images:
            src = os.path.join(org_path, cp_test_images)
            dst = os.path.join(test_dirPath, cp_test_images)
            shutil.copyfile(src, dst)
        for cp_predict_images in predict_images:
            src = os.path.join(org_path, cp_predict_images)
            dst = os.path.join(predict_dirPath, cp_predict_images)
            shutil.copyfile(src, dst)


if __name__ == '__main__':

    # Splitting images into 3 datasets
    data_splitting()

