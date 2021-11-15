# CarND-pj4_Behavioral_Cloning
### Udacity Self-Driving Car Engineer - Project4: Behavioral Cloning (Computer Vision | Deep Learning: end-to-end convolutional networks)

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)



The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road

[//]: # (Image References)

[image1]: ./examples/nvidia-net.png "Nvidia Net"
[image2]: ./examples/sides_image.png "Side Images"


---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.ipynb containing the script to create and train the model.
* model.h5 containing a trained convolution neural network.
* drive.py for driving the car in autonomous mode.
* video.py for recording the simulation. 


Simulator: https://github.com/udacity/self-driving-car-sim

install: socketio and engineio
```sh
pip install python-socketio==4.6.0
pip install python-engineio==3.13.0
```
Drive
```sh
python drive.py model.h5
```
recording
```sh
python video.py run1
(python video.py run1 --fps 48)
```

The model.ipynb file contains the code for training the convolution neural network and saving the model. The file shows the pipeline that I used for training and validating the model, and it contains comments to explain how the code works.

## Model Architecture and Training Strategy

My model is based on [Nvidia End to End Learning for Self-Driving Cars](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) architecture. I thought this architecture might be appropriate because it has been proven to work in real life and because of its size it is suitable for real-time applications. The architecture is showing in the following image:

<p align="center"><img src="./misc/nvidia-net.png" alt="nvidia-net" width="500" class="center"/></p>

The model includes RELU activation functions to introduce nonlinearity. The data is preprocessed in the first two layers. First I apply a crop over the image to get an interesting area:
```sh
 model.add(Cropping2D(cropping=((60,20), (0,0)), input_shape=(160,320,3)))
```
Then normalize the data to a value in the range [-1,1]:
```sh
model.add(Lambda(lambda x: x/127.5 - 1.))
```

The model also contains dropout layers in order to reduce overfitting. The activation rate is set in 0.5.


#### 3. Appropriate training data

The model used an adam optimizer and the batch size is set to 256.

I use different set of training data:
    - Record driving clockwise
    - Record driving counterclockwise
    - Record recovering from the left and right sides
    - Record recovering from the left and right curves
    - Added udacity data to expand the dataset

Total data was: 42927 images.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I started testing with just my records driving normally clockwise and counter clockwise. It worked when the car was in the center of the lane, but he doesn't know how to recover when he gets close to the sidelines.

Then I added recovery data and udacity to expand a dataset and the behavior improved a lot. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. To combat the overfitting, I modified the model added dropout layers.


## 2. Data augmentation

I use sides images recorded in every test to augment the dataset. I modified the angle for these images by adding a correction factor of 0.25. Here is an example of how it looks:

<p align="center"><img src="./misc/sides_image.png" alt="sides_image" class="center"/></p>

I also flipped half of images and angles thinking that this would balance the data between lines on the left and right side. Moreover I changed the contrast of half the images to help to generalize the model. The data set is randomly shuffled after each itration.