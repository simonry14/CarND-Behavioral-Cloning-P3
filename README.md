
**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/nvidia-cnn.png "Model Visualization"
[image2]: ./examples/normal.jpg "Normal Image"
[image3]: ./examples/flipped.jpg "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md summarizing the results
* video.mp4 video of vehicle being driven autonomously by the model


#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with a normalisation layer (Keras Lambda Layer), 5 convolution layers and 3 fully connected layers. The convolution layers have filter sizes of 5 and 3.

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (model.py line 68).

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py line 75).

The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py lines 83 - 85). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 83).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. To recover from the left and right sides of the road, left and right images were utilised with appropriate correction values for the sterring angles. Furthermore to mitigate left turn bias, images were flipped and corresponding steering angles multiplied by -1.

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My overall strategy for deriving a good model was to use the Nvidia Model architecture since it has been proven to be very successful in real self-driving car tasks.

To combat the overfitting, I modified the model and added a dropout layer with probability of 0.4


#### 2. Final Model Architecture

The final model architecture (model.py lines 66-80) consists of a convolution neural network with a normalisation layer (Keras Lambda Layer), 5 convolution layers and 3 fully connected layers. The convolution layers have filter sizes of 5 and 3. The model includes RELU layers to introduce nonlinearity.

Here is a visualization of the architecture (Source: NVIDIA End-to-End Deep Learning for Self-Driving Cars)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

I was unable to create my own training data online as the simulator wasn't very responsive in training mode even when the Screen Resolution and Graphics Quality was set to the least posibble values. I downloaded the Udacity simulator and ran it on my local machine and collected some data which I uploaded to the Google Drive and imported into the project workspace. Given that the data is cleared whenever the classroom is closed I was unable to use it So I ended up using the data provided by Udacity.


To augment the data set, I  flipped images and took the opposite sign of steering angles in order to mitigate the left turn bias. The left turn bias is inherent in the training data set as the track moves anti-clockwise and so there are only left turns. Here is an example of a center camera image and its flipped counterpart.

![alt text][image2]
![alt text][image3]

To further augment the data, the left and right camera images were also fed to the model as if they were coming from the center camera. This way, the model was taught  how to steer if the car drifts off to the left or the right of the track. A correction parameter of +0.4 was used on the left camera image while -0.3 was used on the right camera image.


The data was then preprocessed by utilising a keras lambda layer to parallelize image normalization.


I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 2 as the training and validation loses didnt improve past epoch 2. I used an adam optimizer so that manually training the learning rate wasn't necessary.
