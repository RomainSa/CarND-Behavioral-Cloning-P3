from keras.models import Sequential
from keras.layers import Lambda
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.advanced_activations import ELU
from keras.optimizers import Adam

import sys
import socket
import os
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import utils
import parameters

# hyperparameters
side_adjustment = 0.20
angle_adjustment = 0.10
p_zeros_samples_to_exclude = 0.95
p_near_zeros_samples_to_exclude = 0.75
args = sys.argv
if len(args) == 5:
    side_adjustment = float(args[1])
    angle_adjustment = float(args[2])
    p_zeros_samples_to_exclude = float(args[3])
    p_near_zeros_samples_to_exclude = float(args[4])

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
    if 'recovering_from_left' in destination_folder.lower():
        # for recovering from left data we only keep sharp right turns (center and left images)
        min_angle = 0.10
        mask = (y > min_angle)
        speed = speed[mask]
        paths = paths[mask]
        y = y[mask]
    if 'left_side_driving' in destination_folder.lower():
        y += side_adjustment
    if 'right_side_driving' in destination_folder.lower():
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

# exclude samples that are exactly or near 0
if p_zeros_samples_to_exclude > 0 or p_near_zeros_samples_to_exclude:
    near_zeros_examples = np.where((np.abs(y) < 0.30) & (np.abs(y) > 0))[0]
    zeros_examples = np.unique(np.concatenate((np.where(y == 0)[0],
                                               np.where(np.abs(y) == angle_adjustment)[0])))
    near_zeros_samples_to_exclude = np.random.choice(near_zeros_examples,
                                                     int(p_near_zeros_samples_to_exclude * near_zeros_examples.shape[0]),
                                                     False)
    zeros_samples_to_exclude = np.random.choice(zeros_examples,
                                                int(p_zeros_samples_to_exclude * zeros_examples.shape[0]),
                                                False)
    indexes = np.array([i for i in range(y.shape[0]) if (i not in zeros_samples_to_exclude
                                                         and i not in near_zeros_samples_to_exclude)])
    speed = speed[indexes]
    paths = paths[indexes]
    y = y[indexes]

# split data into train / validation / test
samples = [(p_, y_) for p_, y_ in zip(paths, y)]
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# test data
_, y_test, paths_test, speed_test = utils.load_data(data_folder + 'track2/', return_images=False)
mask = np.array([parameters.center_images_pattern in p for p in paths_test])
y_test = y_test[mask]
paths_test = paths_test[mask]
speed_test = speed_test[mask]
test_samples = [(p_, y_) for p_, y_ in zip(paths_test, y_test)]

"""
based on 'End to End Learning for Self-Driving Cars' by Nvidia
"""

model = Sequential()

# cropping layer
model.add(Cropping2D(cropping=((60, 25), (0, 0)), input_shape=(160, 320, 3)))

# normalization layer
model.add(Lambda(lambda x: (x / 255.0) - 0.5))

# convolution layers
model.add(Convolution2D(input_shape=(160, 320, 3), nb_filter=24, nb_row=5, nb_col=5, subsample=(2, 2),
                        border_mode='valid'))
model.add(ELU())
model.add(Dropout(0.50))

model.add(Convolution2D(nb_filter=36, nb_row=5, nb_col=5, subsample=(2, 2), border_mode='valid'))
model.add(ELU())
model.add(Dropout(0.50))

model.add(Convolution2D(nb_filter=48, nb_row=5, nb_col=5, subsample=(2, 2), border_mode='valid'))
model.add(ELU())
model.add(Dropout(0.50))

model.add(Convolution2D(nb_filter=64, nb_row=3, nb_col=3, subsample=(1, 1), border_mode='valid'))
model.add(ELU())
model.add(Dropout(0.50))

model.add(Convolution2D(nb_filter=64, nb_row=3, nb_col=3, subsample=(1, 1), border_mode='valid'))
model.add(ELU())
model.add(Dropout(0.50))

# fully connected layers
model.add(Flatten())
model.add(Dense(100))
model.add(ELU())
model.add(Dropout(0.50))

model.add(Dense(50))
model.add(ELU())
model.add(Dropout(0.50))

model.add(Dense(10))
model.add(ELU())

model.add(Dense(1))


# data generators
def generator(samples_, batch_size=64):
    num_samples = len(samples_)
    while True:
        samples_ = sklearn.utils.shuffle(samples_)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples_[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                name = batch_sample[0]
                # randomly flip data
                flip = np.random.rand() > 0.5
                image = utils.load_images([name], flip)
                images.append(image)
                angle = -batch_sample[1] if flip else batch_sample[1]
                angles.append(angle)
            # trim image to only see section with road
            X_ = np.array(images)
            y_ = np.array(angles)
            yield X_, y_

train_generator = generator(train_samples, batch_size=64)
validation_generator = generator(validation_samples, batch_size=64)
test_generator = generator(test_samples, batch_size=64)

# compile, train and save the model
adam_ = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer=adam_, loss='mean_squared_error')
model.fit_generator(train_generator, nb_epoch=10, samples_per_epoch=len(train_samples),
                    validation_data=validation_generator, nb_val_samples=len(validation_samples))
evaluation = model.evaluate_generator(test_generator, val_samples=len(test_samples))
print('\nTest MSE: {}\n'.format(evaluation))
model_name = 'model_SA' + str(side_adjustment) + '_AA' + str(angle_adjustment) + '_Z' +\
             str(p_zeros_samples_to_exclude) + '_NZ' + str(p_near_zeros_samples_to_exclude)
model.save(model_name)
