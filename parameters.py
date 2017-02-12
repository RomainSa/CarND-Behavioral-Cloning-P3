import numpy as np

# hostname
local_hostname = 'MacBook-Pro-de-Romain.local'

# folders
local_data_folder = '/Users/roms/GitHub/SDCND_T1_Simulator/data/'
remote_data_folder = '/run/user/1001/data/'
udacitydata_folder = 'udacity/'
mydata_folder = 'mydata/'
images_folder = 'IMG/'

# file names
steering_filename = 'driving_log.csv'
left_images_pattern = 'IMG/left_'
right_images_pattern = 'IMG/right_'
steering_variables = np.array(['img_center', 'img_left', 'img_right', 'steering_angle', 'throttle', 'brake', 'speed'])
