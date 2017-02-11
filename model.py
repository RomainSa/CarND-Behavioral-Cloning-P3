import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D


def load_images(paths):
    """
    Loads images given their paths and return a numpy array
    """
    X = []
    for img in paths:
        X.append(np.asarray(Image.open(img)))
    X = np.array(X)
    return X


# data paths - normal driving and recovering from side
data_driving = '/Users/roms/GitHub/SDCND_T1_Simulator/data/Smooth_driving/'
data_recovering = '/Users/roms/GitHub/SDCND_T1_Simulator/data/Recovering_from_left/'
images_folder = 'IMG/'
steering_file = 'driving_log.csv'
steering_variables = np.array(['img_center', 'img_left', 'img_right', 'steering_angle', 'throttle', 'brake', 'speed'])

# loading data
file1 = data_driving + steering_file
paths1 = np.genfromtxt(file1, dtype='str', delimiter=',')[:, 0]
y1 = np.genfromtxt(file1, dtype=float, delimiter=',')[:, 3]
X1 = load_images(paths1)

file2 = data_recovering + steering_file
paths2 = np.genfromtxt(file2, dtype='str', delimiter=',')[:, 0]
y2 = np.genfromtxt(file2, dtype=float, delimiter=',')[:, 3]
X2 = load_images(paths1)

# filtering images from recovering data - using only right turns (recovering turns)
X2 = X2[y2 > 0]
y2 = y2[y2 > 0]

# data merge
X = np.concatenate((X1, X2))
y = np.concatenate((y1, y2))

# input cropping
X = X[:, 50:130, :, :]

# input augmentation: horizontal flipping
X = np.concatenate((X, X[:, :, ::-1, :]))
y = np.concatenate((y, -y))


# TODO: input augmentation: brightness change


# TODO: input augmentation: color change


# TODO: input augmentation: distribution adjustment, to have a uniform one


# input normalization
X = (X-255/2.)/255.

X.max()
X.min()

# input shuffle
X, y = shuffle(X, y)

# based on 'End to End Learning for Self-Driving Cars' by Nvidia
model = Sequential()

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

model.add(Flatten())

model.add(Dense(100))
model.add(Activation('relu'))

model.add(Dense(50))
model.add(Activation('relu'))

model.add(Dense(10))
model.add(Activation('relu'))

model.add(Dense(1))
model.add(Activation('softmax'))

# compile and train the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X, y, batch_size=32, nb_epoch=2, validation_split=0.2)
