from shutil import move, rmtree
import urllib.request
import zipfile
import numpy as np
from PIL import Image
import parameters


def download_and_unzip(url, extraction_folder, extracted_folder_name):
    """
    Download and unzip a zip file, given an url
    """
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


def load_data(data_folder, return_images=True):
    """
    Loads simulator training data, given a folder
    """
    steeringfile_path = data_folder + parameters.steering_filename
    images_paths = np.genfromtxt(steeringfile_path, skip_header=1, dtype='str', delimiter=',')[:, :3].flatten()
    images_paths = np.array([data_folder + p.strip() for p in images_paths])
    y = np.repeat(np.genfromtxt(steeringfile_path, skip_header=1, dtype=float, delimiter=',')[:, 3], 3)
    speed = np.repeat(np.genfromtxt(steeringfile_path, skip_header=1, dtype=float, delimiter=',')[:, 6], 3)
    if return_images:
        X = load_images(images_paths)
        return X, y, images_paths, speed
    else:
        return None, y, images_paths, speed


def shift_image_vertically(img, vertical_shift):
    """
    Adds a vertical shift to an image
    """
    img_ = img.copy()
    if vertical_shift > 0:
        img_[vertical_shift:, :, :] = img_[:-vertical_shift, :, :]
        img_[:vertical_shift, :, :] = 0
    elif vertical_shift < 0:
        img_[:vertical_shift, :, :] = img_[-vertical_shift:, :, :]
        img_[vertical_shift:, :, :] = 0
    return img_


def change_brightness(img, brightness_change):
    """
    Performs image brightness change by converting it to YCrBr then back to RGB
    """
    img2 = np.asarray(Image.fromarray(img).convert('YCbCr')).astype(np.float32)
    img2[:, :, 0] += -brightness_change
    img2[img2 > 255] = 255
    img2[img2 < 0] = 0
    return np.asarray(Image.fromarray(img2.astype(np.uint8), mode='YCbCr').convert('RGB'))


def load_images(paths, flip=False, change_brightness=False):
    """
    Loads images given their paths and return a numpy array
    """
    X = []
    if isinstance(paths, str):
        paths = [paths]
    for img in paths:
        x = np.asarray(Image.open(img))
        # random brightness change
        if change_brightness:
            brightness_change = np.random.randint(-50, +50)
            x = change_brightness(x, brightness_change)
        # flip image
        if flip:
            x = x[:, ::-1, :]
        X.append(x)
    if len(X) > 1:
        X = np.array(X)
    else:
        X = X[0]
    return X
