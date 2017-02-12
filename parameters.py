import numpy as np

# hostname
local_hostname = 'MacBook-Pro-de-Romain.local'

# folders
local_data_folder = '/Users/roms/GitHub/SDCND_T1_Simulator/data/'
remote_data_folder = '/run/user/1001/data/'
data_folders_list = ['udacity/', 'Smooth_driving/', 'Recovering_from_left/']
default_images_folder = 'IMG/'

# file names
steering_filename = 'driving_log.csv'
left_images_pattern = 'IMG/left_'
right_images_pattern = 'IMG/right_'
steering_variables = np.array(['img_center', 'img_left', 'img_right', 'steering_angle', 'throttle', 'brake', 'speed'])

# urls
urls_list = ['https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip',
             'https://s3-us-west-2.amazonaws.com/carnd-rs/data.zip',
             'https://s3-us-west-2.amazonaws.com/carnd-rs/data.zip']
