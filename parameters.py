import numpy as np

# hostname
local_hostname = 'MacBook-Pro-de-Romain.local'

# folders
local_data_folder = '/Users/roms/GitHub/SDCND_T1_Simulator/data/'
remote_data_folder = '/home/carnd//data/'
data_folders_list = ['Smooth_driving3/',  'Left_side_driving2/', 'Right_side_driving2/']
default_images_folder = 'IMG/'

# file names
steering_filename = 'driving_log.csv'
center_images_pattern = 'IMG/center_'
left_images_pattern = 'IMG/left_'
right_images_pattern = 'IMG/right_'
steering_variables = np.array(['img_center', 'img_left', 'img_right', 'steering_angle', 'throttle', 'brake', 'speed'])

# urls
urls_list = ['https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip',
             'https://s3-us-west-2.amazonaws.com/carnd-rs/Left_side_driving.zip',
             'https://s3-us-west-2.amazonaws.com/carnd-rs/Right_side_driving.zip']
