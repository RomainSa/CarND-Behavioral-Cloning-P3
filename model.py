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
        min_angle = 0.10
        mask = (y > min_angle)
        speed = speed[mask]
        paths = paths[mask]
        y = y[mask]
    side_adjustment = 0.15
    if destination_folder == 'Left_side_driving/':
        y += side_adjustment
    if destination_folder == 'Right_side_driving/':
        y -= side_adjustment
    speed_list.append(speed)
    paths_list.append(paths)
    y_list.append(y)

# concatenate data
speed = np.concatenate(speed_list)
paths = np.concatenate(paths_list)
y = np.concatenate(y_list)

# remove low speed data
min_speed = 20
mask = speed > min_speed
paths = paths[mask]
y = y[mask]
speed = speed[mask]

# right and left cameras angle adjustment
angle_adjustment = 0.05
if angle_adjustment > 0:
    left_images = np.array([parameters.left_images_pattern in p for p in paths])
    right_images = np.array([parameters.right_images_pattern in p for p in paths])
    y[left_images] += angle_adjustment
    y[right_images] -= angle_adjustment
else:
    center_images = np.array([parameters.center_images_pattern in p for p in paths])
    paths = paths[center_images]
    y = y[center_images]
    speed = speed[center_images]

# flips some data horizontally
mask = (y != 0)
speed = np.concatenate((speed, speed[mask]))
paths = np.concatenate((paths, paths[mask]))
flip = np.concatenate((np.repeat(False, y.shape[0]), np.repeat(True, mask.sum())))
y = np.concatenate((y, -y[mask]))

# exclude samples that are exactly or near 0
p_zeros_samples_to_exclude = 0.50
p_near_zeros_samples_to_exclude = 0.50
if p_zeros_samples_to_exclude > 0 or p_near_zeros_samples_to_exclude:
    near_zeros_examples = np.where((np.abs(y) < 0.2) & (np.abs(y) > 0))[0]
    zeros_examples = np.unique(np.concatenate((np.where(y == 0)[0],
                                               np.where(np.abs(y) == angle_adjustment)[0])))
    near_zeros_samples_to_exclude = np.random.choice(near_zeros_examples,
                                                     int(p_near_zeros_samples_to_exclude * near_zeros_examples.shape[0]),
                                                     False)
    zeros_samples_to_exclude = np.random.choice(zeros_examples,
                                                int(p_zeros_samples_to_exclude * zeros_examples.shape[0]),
                                                False)
    indexes = np.array([i for i in range(y.shape[0]) if (i not in zeros_samples_to_exclude and i not in near_zeros_samples_to_exclude)])
    speed = speed[indexes]
    paths = paths[indexes]
    y = y[indexes]
    flip = flip[indexes]

# load images data
X = utils.load_images(paths)

# flips it if needed
X[flip] = X[flip][:, :, ::-1, :]

# shuffle data
X, y = shuffle(X, y)


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
model.add(Convolution2D(nb_filter=64, nb_row=3, nb_col=3, subsample=(1, 1), border_mode='valid'))

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
history = model.fit(X, y, batch_size=64, nb_epoch=5, validation_split=0.2)
model.save('model.h5')
