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

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted 

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

###Code Quality

####1. Submssion includes functional code
Using the last version of the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
Please note that I modified the drive.py file so that the car drives at full speed.

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
- 3 fully-connected layers (1164, 100 and 10 neurons respectivelly)

The model includes ELU layers to introduce nonlinearity.

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines ???). 
The model was trained and validated on different data sets to ensure that the model was not overfitting (code lines ???).  
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line ???).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road.  
I used a combination of 3 datasets that I recorded on my own:
- center lane driving: 5 laps driving on the center of the road
- left lane driving: 2 laps driving alongside the left lane of the road
- right lane driving: 2 laps driving alongside the right lane of the road

The right and left lane datasets are used as recovery data.
For details about how I created the training data, see the next section. 
















###Model Architecture and Training Strategy

Is the solution design documented?
Is the model architecture documented?
Is the creation of the training dataset and training process documented?

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to start from [Nvidia's model](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf).
 
???

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

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.












###Simulation

As can be seen on the video file video.mp4 the car is able to navigate correctly on the first track.


