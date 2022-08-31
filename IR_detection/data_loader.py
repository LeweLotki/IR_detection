import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn import preprocessing
from data_prep import find_dirs, find_files

def get_data(path):
    
    data_set = []
    target_set = []
    data_size = 0
    for filedir in find_dirs(path, '*'):
        dir_path = os.path.join(path, filedir)
        for filename in find_files(dir_path, '*'):
            infilename = os.path.join(filedir, filename)
            if not os.path.isfile(infilename): continue
            target = np.zeros((1, 1))
            if int(filedir[-1]) == 1: target[0] = 1
            data_size += 1
            im = cv2.imread(infilename, 0)
            data_set.append(im)
            target_set.append(target)
    data_set = np.reshape(data_set, (data_size, 1, 128, 128)).astype('float32')
    target_set = np.reshape(target_set, (data_size, 1))
    target_set = target_set.T
    
    return data_set, target_set

paths = [
    'data\\train',
    'data\\test'
]
dirname = os.path.dirname(__file__)
for index, path in enumerate(paths):
    paths[index] = os.path.join(dirname, path)

training_data, training_targets = get_data(paths[0])
testing_data, testing_tagrets = get_data(paths[1])