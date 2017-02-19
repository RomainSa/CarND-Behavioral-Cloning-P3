from keras.models import Sequential
from keras.layers import Lambda
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.optimizers import Adam

import socket
import os
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import truncnorm
from sklearn.utils import shuffle
import utils
import parameters


# check if running on local or remote
if socket.gethostname() == parameters.local_hostname:
    remote = False
else:
    remote = True

# get corresponding data directory
if remote:
    data_folder = parameters.remote_data_folder
else:
    data_folder = parameters.local_data_folder
if not os.path.isdir(data_folder):
    os.mkdir(data_folder)

# download data if needed then loads it
speed_list = []
paths_list = []
y_list = []
for destination_folder, url in zip(parameters.data_folders_list, parameters.urls_list):
    if not os.path.isdir(data_folder + destination_folder):
        utils.download_and_unzip(url, data_folder, destination_folder)
    _, y, paths, speed = utils.load_data(data_folder + destination_folder, return_images=False)
    if destination_folder == 'Recovering_from_left2/':
        # for recovering from left data we only keep sharp right turns (center and left images)
        min_angle = 0.25
        mask = (y > min_angle) & np.array([parameters.right_images_pattern not in p for p in paths])
        speed = speed[mask]
        paths = paths[mask]
        y = y[mask]
    speed_list.append(speed)
    paths_list.append(paths)
    y_list.append(y)

# concatenate data
speed = np.concatenate(speed_list)
paths = np.concatenate(paths_list)
y = np.concatenate(y_list)

# right and left cameras angle adjustment
angle_adjustment = 0.10
if angle_adjustment > 0:
    left_images = np.array([parameters.left_images_pattern in p for p in paths])
    right_images = np.array([parameters.right_images_pattern in p for p in paths])
    y[left_images] += angle_adjustment
    y[right_images] -= angle_adjustment
else:   # retain only center images
    center_images = np.array([parameters.center_images_pattern in p for p in paths])
    speed = speed[center_images]
    paths = paths[center_images]
    y = y[center_images]

# loads samples based on a uniform distribution
n_examples = 5000
y_target = np.random.uniform(low=0., high=0.5, size=n_examples)
distances = np.tile(np.abs(y), n_examples).reshape((n_examples, y.shape[0])) - np.vstack(y_target)
indexes = np.argmin(np.abs(distances), axis=1)
flip = y[indexes] * y_target > 0
distances = None   # to empty memory
speed = speed[indexes]
paths = paths[indexes]
y = y[indexes]

# load images data
X = utils.load_images(paths)

# flips some data horizontally
for i, flip_ in enumerate(flip):
    if flip_:
        X[i] = X[i, :, ::-1, :]
y[flip] *= -1


"""
based on 'End to End Learning for Self-Driving Cars' by Nvidia
"""

model = Sequential()

# cropping layer
model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))

# normalization layer
model.add(Lambda(lambda x: (x / 255.0) - 0.5))

# convolution layers
model.add(Convolution2D(input_shape=(X.shape[1:]), nb_filter=24, nb_row=5, nb_col=5, subsample=(2, 2),
                        border_mode='valid'))
model.add(Dropout(0.50))
model.add(Convolution2D(nb_filter=36, nb_row=5, nb_col=5, subsample=(2, 2), border_mode='valid'))
model.add(Dropout(0.50))
model.add(Convolution2D(nb_filter=48, nb_row=5, nb_col=5, subsample=(2, 2), border_mode='valid'))
model.add(Dropout(0.50))
model.add(Convolution2D(nb_filter=64, nb_row=3, nb_col=3, subsample=(1, 1), border_mode='valid'))
model.add(Dropout(0.50))
model.add(Convolution2D(nb_filter=64, nb_row=3, nb_col=3, subsample=(1, 1), border_mode='valid'))
model.add(Dropout(0.50))

# fully connected layers
model.add(Flatten())
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('relu'))
model.add(Dense(1))

# compile, train and save the model
adam_ = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer=adam_, loss='mean_squared_error')
history = model.fit(X, y, batch_size=32, nb_epoch=10, validation_split=0.2)
model.save('model.h5')
