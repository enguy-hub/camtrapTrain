# camtrapTrain

Easy to use code base for training and using species ID ecological models

## Installing requirements:
  
  ```
  pip install -r requirements.txt
  ```
  
- If there are any errors a quick Google search can help resolve them.
  For errors installing tensorflow I recommend this guide: <https://www.youtube.com/watch?v=r7-WPbx8VuY>

- Once requirements are installed, replace the directories in Train and Test with your own images.

- The structure should be as in the examples, with each directory name being the name of the species, filled with images.

- There is a Train and Test directory. Divide your images approximately 9/10 Train and 1/10 Test for each species.

- Once complete, run

```
python MultiClassTrain.py
```
- The model will train using MobileNetV2 as default. You want to pay attention to the Validation Accuracy at the end of an epoch to see how well you are performing on the provided test set. For a comment by comment description of what the code is doing, you can run the view the Jupyter notebook as well. 
  
- There are numerous options to choose from when training. They are used for example:

- The options include:

  - '--epoch'       A number between 1-500          Number of training epochs
  
  - '--batch_size'  A number between 1-64           Number of images placed on GPU. Reduce if receive memory error
  
  - '--img_size'    A number from 32, 64, 128, 256  Training image size. Reduce if receive memory error
  
  - '--network'   Choose one of the following for training. For details see https://keras.io/applications/. We recommend MobileNetV2 and DenseNet201
  
  - 'Xception'
  
  - 'DenseNet201'
  
  - 'MobileNetV2'
  
  - 'NASNetMobile'
  
  - 'Inception_Resnet_V2'
  
  - 'VGG19'   

- To select from the options, an input example looks something like this:

```
python MutliClassTrain.py --epoch 200 --network DenseNetV2 --batch_size 32
```

Source: https://github.com/Schnei1811/Camera_Trap_Species_Classifier