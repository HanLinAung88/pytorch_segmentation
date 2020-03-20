"""
in this script, we calculate the image per channel mean and standard
deviation in the training set, do not calculate the statistics on the
whole dataset, as per here http://cs231n.github.io/neural-networks-2/#datapre
Based on: https://gist.github.com/jdhao/9a86d4b9e4f79c5330d54de991461fd6

e.g.python extract_mean_std.py ~/coral_andreas/coral-video-identification/pytorch_segmentation/data/Coral/train/images ~/coral_andreas/coral-video-identification/pytorch_segmentation/data/Coral/mean_std_coral.json
"""

import sys
import numpy as np
from os import listdir
from os.path import join, isdir
from glob import glob
import cv2
import json

# number of channels of the dataset image, 3 for color jpg, 1 for grayscale img
# you need to change it to reflect your dataset
CHANNEL_NUM = 3


def cal_dir_stat(root):
    pixel_num = 0 # store all pixel number in the dataset
    channel_sum = np.zeros(CHANNEL_NUM)
    channel_sum_squared = np.zeros(CHANNEL_NUM)

    im_pths = glob(join(root, "*.png"))
    print("Extracting mean and std")
    for path in im_pths:
        im = cv2.imread(path) # image in M*N*CHANNEL_NUM shape, channel in BGR order
        im = im/255.0
        pixel_num += (im.size/CHANNEL_NUM)
        channel_sum += np.sum(im, axis=(0, 1))
        channel_sum_squared += np.sum(np.square(im), axis=(0, 1))
    bgr_mean = channel_sum / pixel_num
    bgr_std = np.sqrt(channel_sum_squared / pixel_num - np.square(bgr_mean))
    
    # change the format from bgr to rgb
    rgb_mean = list(bgr_mean)[::-1]
    rgb_std = list(bgr_std)[::-1]
    
    return rgb_mean, rgb_std

# The script assumes that under train_root, there are separate directories for each class
# of training images.
train_root = sys.argv[1] 
file_name = sys.argv[2]
mean, std = cal_dir_stat(train_root)
mean_std_data = {"mean": mean, "std": std}
with open(file_name, 'w') as f:
    json.dump(mean_std_data, f)
