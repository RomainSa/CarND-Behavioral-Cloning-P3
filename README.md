# Project 3: Use Deep Learning to Clone Driving Behavior

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
This repository contains starting files for P3, Behavioral Cloning.

In this project, you will use what you've learned about deep neural networks and convolutional neural networks to clone driving behavior. You will train, validate and test a model using Keras. The model will output a steering angle to an autonomous vehicle.

We have provided a simulator where you can steer a car around a track for data collection. You'll use image data and steering angles to train a neural network and then use this model to drive the car autonomously around the track.

We also want you to create a detailed writeup of the project. Check out the [writeup template](https://github.com/udacity/CarND-Behavioral-Cloning-P3/blob/master/writeup_template.md) for this project and use it as a starting point for creating your own writeup. The writeup can be either a markdown file or a pdf document.

To meet specifications, the project will require submitting four files: 
* model.py (script used to create and train the model)
* drive.py (script to drive the car - feel free to modify this file)
* model.h5 (a trained Keras model)
* a report writeup file (either markdown or pdf)

Optionally, a video of your vehicle's performance can also be submitted with the project although this is optional. This README file describes how to output the video in the "Details About Files In This Directory" section.

Creating a Great Writeup
---
A great writeup should include the [rubric points](https://review.udacity.com/#!/rubrics/432/view) as well as your description of how you addressed each point.  You should include a detailed description of the code used (with line-number references and code snippets where necessary), and links to other supporting documents or external references.  You should include images in your writeup to demonstrate how your code works with examples.  

All that said, please be concise!  We're not looking for you to write a book here, just a brief description of how you passed each rubric point, and references to the relevant code :). 

You're not required to use markdown for your writeup.  If you use another method please just submit a pdf of your writeup.

The Project
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior 
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab enviroment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

The following resources can be found in this github repository:
* drive.py
* video.py
* writeup_template.md

The simulator can be downloaded from the classroom. In the classroom, we have also provided sample data that you can optionally use to help train your model.

## Details About Files In This Directory

### `drive.py`

Usage of `drive.py` requires you have saved the trained model as an h5 file, i.e. `model.h5`. See the [Keras documentation](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model) for how to create this file using the following command:
```sh
model.save(filepath)
```

Once the model has been saved, it can be used with drive.py using this command:

```sh
python drive.py model.h5
```

The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.

#### Saving a video of the autonomous agent

```sh
python drive.py model.h5 run1
```

The fourth argument `run1` is the directory to save the images seen by the agent to. If the directory already exists it'll be overwritten.

```sh
ls run1

[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_424.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_451.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_477.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_528.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_573.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_618.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_697.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_723.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_749.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_817.jpg
...
```

The image file name is a timestamp when the image image was seen. This information is used by `video.py` to create a chronological video of the agent driving.

### `video.py`

```sh
python video.py run1
```

Create a video based on images found in the `run1` directory. The name of the video will be name of the directory following by `'.mp4'`, so, in this case the video will be `run1.mp4`.

Optionally one can specify the FPS (frames per second) of the video:

```sh
python video.py run1 --fps 48
```

The video will run at 48 FPS. The default FPS is 60.

#### Why create a video

1. It's been noted the simulator might perform differently based on the hardware. So if your model drives succesfully on your machine it might not on another machine (your reviewer). Saving a video is a solid backup in case this happens.
2. You could slightly alter the code in `drive.py` and/or `video.py` to create a video of what your model sees after the image is processed (may be helpful for debugging).

###Model Architecture and Training Strategy

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

The final model architecture (model.py lines 99-143) is based on Nvidia's paper: [End to end learning for self driving cars](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)

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


