import urllib.request
import zipfile
import numpy as np
from PIL import Image
import parameters


def load_data(steeringfile_folder):
    steeringfile_path = steeringfile_folder + parameters.steering_filename
    images_paths = np.genfromtxt(steeringfile_path, skip_header=1, dtype='str', delimiter=',')[:, :3].flatten()
    images_paths = np.array([steeringfile_path + p for p in images_paths])
    y = np.repeat(np.genfromtxt(steeringfile_path, skip_header=1, dtype=float, delimiter=',')[:, 3], 3)
    X = load_images(images_paths)
    return X, y, images_paths


def load_images(paths):
    """
    Loads images given their paths and return a numpy array
    """
    X = []
    for img in paths:
        X.append(np.asarray(Image.open(img)))
    X = np.array(X)
    return X


def download_and_unzip(url, folder):
    file_handle, _ = urllib.request.urlretrieve(url)
    zip_ref = zipfile.ZipFile(file_handle, 'r')
    zip_ref.extractall(folder)
    zip_ref.close()
