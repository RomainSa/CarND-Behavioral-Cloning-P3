import urllib.request
import zipfile
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Lambda
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.optimizers import Adam


def load_images(paths):
    """
    Loads images given their paths and return a numpy array
    """
    X = []
    for img in paths:
        X.append(np.asarray(Image.open(img)))
    X = np.array(X)
    return X


# download udacity data
url = 'https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip'
filehandle, _ = urllib.request.urlretrieve(url)
zip_ref = zipfile.ZipFile(filehandle, 'r')
zip_ref.extractall('')
zip_ref.close()

# data paths - normal driving and recovering from side
data_udacity = 'data/'
images_folder = 'IMG/'
steering_file = 'driving_log.csv'
steering_variables = np.array(['img_center', 'img_left', 'img_right', 'steering_angle', 'throttle', 'brake', 'speed'])

# loading data
file = data_udacity + steering_file
paths = np.genfromtxt(file, skip_header=1, dtype='str', delimiter=',')[:, 0]
prefix = 'data/'
paths = np.array([prefix + p for p in paths])
y = np.genfromtxt(file, skip_header=1, dtype=float, delimiter=',')[:, 3]
X = load_images(paths)

# input augmentation: horizontal flipping
X = np.concatenate((X, X[:, :, ::-1, :]))
y = np.concatenate((y, -y))


# TODO: input augmentation: brightness change


# TODO: input augmentation: color change


# TODO: input augmentation: distribution adjustment, to have a uniform one
#plt.hist(y)

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
history = model.fit(X, y, batch_size=32, nb_epoch=10, validation_split=0.2)
model.save('model.h5')
