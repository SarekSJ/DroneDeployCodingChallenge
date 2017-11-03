import numpy as np
import cv2
from matplotlib import pyplot as plt
import transforms3d
import math
from drawings import *
from Feature import *

# iPhone 6 camera values (Used to create the intrinsic matrix)
focal_length = 4.15
width = 3264
length = 2448
# Divide width by sensor width to get sx value
sx = width / 4.89
# Dive length by sensor length to get sy value
sy = length / 3.67
fx=float(focal_length * sx)
print (3264 / 4.8)
fy=float(focal_length * sy)
print length / 3.6
cx=float(width / 2)
cy=float(length / 2)


# Template Values

# Creating the intrinsic matrix
cam_mat = np.matrix([[fx, 0, cx],
                    [0,  fy, cy],
                    [0,   0,  1]])
base_dir = './Camera Localization/'
files = ['IMG_6719', 'IMG_6720', 'IMG_6721', 'IMG_6722', 'IMG_6723', 'IMG_6724', 'IMG_6725', 'IMG_6726', 'IMG_6727']
extension = '.JPG'
pattern_file = './Camera Localization/pattern.png'

def main():
    for file in files:
        get_pose_and_coordinates(pattern_file, base_dir + file + extension, cam_mat)

if __name__ == '__main__':
    main()