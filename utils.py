from shutil import move, rmtree
import urllib.request
import zipfile
import numpy as np
from PIL import Image
import parameters


def load_data(data_folder, return_images=True):
    steeringfile_path = data_folder + parameters.steering_filename
    images_paths = np.genfromtxt(steeringfile_path, skip_header=1, dtype='str', delimiter=',')[:, :3].flatten()
    images_paths = np.array([data_folder + p.strip() for p in images_paths])
    y = np.repeat(np.genfromtxt(steeringfile_path, skip_header=1, dtype=float, delimiter=',')[:, 3], 3)
    if return_images:
        X = load_images(images_paths)
        return X, y, images_paths
    else:
        return None, y, images_paths


def load_images(paths):
    """
    Loads images given their paths and return a numpy array
    """
    X = []
    for img in paths:
        X.append(np.asarray(Image.open(img).convert('LA')))
    X = np.array(X)
    return X


def download_and_unzip(url, extraction_folder, extracted_folder_name):
    file_handle, _ = urllib.request.urlretrieve(url)
    zip_ref = zipfile.ZipFile(file_handle, 'r')
    zip_ref.extractall(extraction_folder)
    zip_ref.close()
    if extracted_folder_name is not None and extracted_folder_name != 'data':
        try:
            rmtree(extraction_folder + extracted_folder_name)
        except:
            pass
        move(extraction_folder + 'data/', extraction_folder + extracted_folder_name)
