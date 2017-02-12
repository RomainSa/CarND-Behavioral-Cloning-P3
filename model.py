from shutil import move
import socket
import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Lambda
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.optimizers import Adam
import utils
import parameters


# parameters
if socket.gethostname() == parameters.local_hostname:
    remote = False
else:
    remote = True

# get data directory
if remote:
    data_folder = parameters.remote_data_folder
    try:
        os.mkdir(data_folder, data_folder)
    except:
        pass
else:
    data_folder = parameters.local_data_folder

# download data if remote
if remote:
    # Udacity data
    url_udacity = 'https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip'
    utils.download_and_unzip(url_udacity, data_folder)
    move(data_folder + 'data', data_folder + parameters.udacitydata_folder)
    # collected data
    url_mydata = 'https://s3-us-west-2.amazonaws.com/carnd-rs/data.zip'
    utils.download_and_unzip(url_mydata, data_folder)
    move(data_folder + 'data', data_folder + parameters.mydata_folder)

# loading data
subfolder0 = ''
X0, y0, paths0 = utils.load_data(data_folder + parameters.udacitydata_folder + subfolder0)

subfolder1 = 'Smooth_driving/'
X1, y1, paths1 = utils.load_data(data_folder + parameters.mydata_folder + subfolder1)

subfolder2 = 'Recovering_from_left/'
X2, y2, paths2 = utils.load_data(data_folder + parameters.mydata_folder + subfolder2)
paths2 = paths2[y2 > 0]
X2 = X2[y2 > 0]
y2 = y2[y2 > 0]

# concatenate data
X = np.concatenate((X0, X1, X2))
y = np.concatenate((y0, y1, y2))
paths = np.concatenate((paths0, paths1, paths2))

# right and left cameras angle adjustment
angle_adjustment = 0.05
left_images = np.array([parameters.left_images_pattern in p for p in paths])
right_images = np.array([parameters.right_images_pattern in p for p in paths])
y[left_images] += angle_adjustment
y[right_images] -= angle_adjustment

# input augmentation using horizontal flipping
X = np.concatenate((X, X[:, :, ::-1, :]))
y = np.concatenate((y, -y))


# TODO: input augmentation: brightness change


# TODO: input augmentation: color change


# TODO: input augmentation: distribution adjustment, to have a uniform one
#plt.hist(y0)

# input shuffle
X, y = shuffle(X, y)


"""
based on 'End to End Learning for Self-Driving Cars' by Nvidia
"""

model = Sequential()

# cropping layer
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160, 320, 3)))

# normalization layer
model.add(Lambda(lambda x: (x / 255.0) - 0.5))

# convolution layers
model.add(Convolution2D(input_shape=(X.shape[1:]), nb_filter=24, nb_row=5, nb_col=5, subsample=(2, 2), border_mode='valid'))
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
model.add(Dense(100), activation='relu')
model.add(Dense(50), activation='relu')
model.add(Dense(10), activation='relu')
model.add(Dense(1))

# compile, train and save the model
adam_ = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer=adam_, loss='mean_squared_error')
history = model.fit(X, y, batch_size=32, nb_epoch=20, validation_split=0.2)
model.save('model.h5')
