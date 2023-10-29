# Importing necessary packages
import os


# Initialize path to 'original' folder that is containing the original images
original_data_dir = 'org_img'

# Get all classes and put them in an array
class_names = [item for item in os.listdir(original_data_dir)]

# Define the names of the train and validation directories
train = 'Train'
test = 'Test'
predict = 'Predict'

# Path to train, validate, and predict directories
train_dir = os.path.join(train)
os.mkdir(train_dir) if not os.path.isdir(train_dir) else None
test_dir = os.path.join(test)
os.mkdir(test_dir) if not os.path.isdir(test_dir) else None
predict_dir = os.path.join(predict)
os.mkdir(predict_dir) if not os.path.isdir(predict_dir) else None

# Initialize the output directory, where the extracted
# features (in CSV file format) will be stored
# base_csv_path = "output"
# os.mkdir(base_csv_path) if not os.path.isdir(base_csv_path) else None



