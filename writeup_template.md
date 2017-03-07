#**Behavioral Cloning** 

##Writeup Template

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/center.png "Center image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted 

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* parameters.py containing configuration parameters
* utils.py containing useful functions

###Code Quality

####1. Submssion includes functional code
Using the last version of the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
Please note that I modified the drive.py file so that the car drives at speed 20.

####2. Submssion code is usable and readable

The model.py file contains the code for training and saving the convolution neural network (which is saved under the name model.h5).  
The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model is based on Nvidia's paper: [End to end learning for self driving cars](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)

It consists of:
- 1 cropping layer (using Keras Cropping2D layer)
- 1 normalization layer (using Keras lambda layer) so that each data point is between -0.5 and 0.5
- 5 convolutional layers (3 with 5x5 kernels and between 24 to 48 filters and 2 with 3x3 kernels and 64 filters)
- 3 fully-connected layers (100, 50 and 10 neurons respectivelly)

The model includes ELU layers to introduce nonlinearity (lines 110-140).

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 111-141). 
The model was trained and validated on different data sets to ensure that the model was not overfitting (code lines 168-170).  
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 173).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road.  
I used Udacity data, which contains a few laps on the training track
For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

Is the solution design documented?
Is the model architecture documented?
Is the creation of the training dataset and training process documented?

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to start from [Nvidia's model](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf).
I thought this model might be appropriate because in the paper, the author use an analog strategy to drive a car using only images as inputs.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set.
One of the parameters that was crucial to change was the initial learning rate of the Adam optimizer.
I noticed that it was necessary to use a low starting learning rate (10-3). 

Another important step was to flip the data in order to add more images.

I also ended up reduced the number of neurons in the fully connected layers compared to Nvidia's fully connected layers (1164, 100 and 10 neurons respectively).

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines ???) is based on Nvidia's paper: [End to end learning for self driving cars](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)

It consists of the following layers:

| Layer           | Details                                            |
|-----------------|----------------------------------------------------|
| Input           | 160x320 RGB image                                  |
| Cropping        | 60 pixels on top, 25 pixels on bottom (first axis) |
| Normalization   | so that each data point is between -0.5 and +0.5   |
| Convolution     | 24 filters, 5x5 kernel                             |
| Activation      | ELU                                                |
| Dropout         | 50% of neurons are zeroed                          |
| Convolution     | 36 filters, 5x5 kernel                             |
| Activation      | ELU                                                |
| Dropout         | 50% of neurons are zeroed                          |
| Convolution     | 48 filters, 5x5 kernel                             |
| Activation      | ELU                                                |
| Dropout         | 50% of neurons are zeroed                          |
| Convolution     | 64 filters, 3x3 kernel                             |
| Activation      | ELU                                                |
| Dropout         | 50% of neurons are zeroed                          |
| Convolution     | 64 filters, 3x3 kernel                             |
| Activation      | ELU                                                |
| Dropout         | 50% of neurons are zeroed                          |
| Fully connected | 100 neurons                                        |
| Activation      | ELU                                                |
| Dropout         | 50% of neurons are zeroed                          |
| Fully connected | 50 neurons                                         |
| Activation      | ELU                                                |
| Dropout         | 50% of neurons are zeroed                          |
| Fully connected | 10 neurons                                         |
| Activation      | ELU                                                |
| Dropout         | 50% of neurons are zeroed                          |
| Output          | scalar                                             |

As we can see, the model includes ELU layers to introduce nonlinearity and dropout to reduce overfitting.

Here is a visualization of Nvidia's model architecture:
![alt text][nvidia_architecture]

####3. Creation of the Training Set & Training Process

I have made several attemps to create my own datasets using my mouse to steer the car.
Yet, the models trained using this data where not as good as those trained using udacity data.
In the end, I ended up using only Udacity's dataset.

This dataset consist of 
Here is an example image of center lane driving:

![alt text][image1]

To augment the data sat, I also flipped images and angles thinking that this would reduce bias by making the steering distribution.

After the collection process, I had X number of data points.
I then preprocessed this data by cropping 60 pixels at the top of the image and 25 pixels at the bottom. The goal of this cropping was to 
prevent the model from overfitting by learning from the sky or trees for example.

I finally randomly shuffled the data set and put 10% of the data into a validation set and 10% in a test set. 

I used this training data for training the model.
The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 20.
I used an adam optimizer so that manually training the learning rate wasn't necessary.

###Simulation

As can be seen on the video file video.mp4 the car is able to navigate correctly on the first track.
