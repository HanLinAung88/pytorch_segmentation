from base import BaseDataSet, BaseDataLoader
from utils import palette
import numpy as np
import os
import torch
import cv2
import json
from PIL import Image
from glob import glob
from torch.utils.data import Dataset
from torchvision import transforms


class CoralDataset(BaseDataSet):
    """
	Coral dataset
    """
    def __init__(self, **kwargs):
        self.num_classes = 2
        super(CoralDataset, self).__init__(**kwargs)

    def _set_files(self):
        if self.split in  ["train", "test"]:
            self.image_dir = os.path.join(self.root, self.split, 'images')
            self.label_dir = os.path.join(self.root, self.split, 'masks_ch1')
            self.label_dir_2 = os.path.join(self.root, self.split, 'masks_ch4')
            self.files = [os.path.basename(path).split('.')[0] for path in glob(self.image_dir + '/*.png')]
        else: raise ValueError(f"Invalid split name {self.split}")
    
    def _load_data(self, index):
        image_id = self.files[index]
        image_path = os.path.join(self.image_dir, image_id + '.png')
		# There are two corals we are detecting
        label_path = os.path.join(self.label_dir, image_id + '_mask.png')
        label_path_2 = os.path.join(self.label_dir_2, image_id + '_mask.png')

        image = np.asarray(Image.open(image_path).convert('RGB'), dtype=np.float32)
        label = np.asarray(Image.open(label_path), dtype=np.int32)  
        label_2 = np.asarray(Image.open(label_path_2), dtype=np.int32)  

        final_label = np.where(label != 0, label, label_2)

        final_label[final_label == 255] = 1
        return image, final_label, image_id

class Coral(BaseDataLoader):
    def __init__(self, data_dir, batch_size, split, crop_size=None, base_size=None, scale=True, num_workers=1, val=False,
                    shuffle=False, flip=False, rotate=False, blur= False, augment=False, val_split= None, return_id=False):

        # RUN data/extract_mean_std.py whenever you have a new dataset to update mean and std
        with open(os.path.join(data_dir, 'mean_std_coral.json')) as f:
            mean_std = json.load(f)
          
        self.MEAN = mean_std["mean"] 
        self.STD = mean_std["std"] 

        kwargs = {
            'root': data_dir,
            'split': split,
            'mean': self.MEAN,
            'std': self.STD,
            'augment': augment,
            'crop_size': crop_size,
            'base_size': base_size,
            'scale': scale,
            'flip': flip,
            'blur': blur,
            'rotate': rotate,
            'return_id': return_id,
            'val': val
        }

        self.dataset = CoralDataset(**kwargs)
        super(Coral, self).__init__(self.dataset, batch_size, shuffle, num_workers, val_split)
