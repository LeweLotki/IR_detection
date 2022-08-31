import os, fnmatch, sys, cv2
from pathlib import Path

def find_dirs(directory, pattern):
    for item in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, item)):
            if fnmatch.fnmatch(item, pattern):
                filename = os.path.join(directory, item)
                yield filename


def find_files(directory, pattern):
    for item in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, item)):
            if fnmatch.fnmatch(item, pattern):
                filename = os.path.join(directory, item)
                yield filename

def prepering_data(infilename):
    
        print(filename)
        im = cv2.imread(infilename)
        gray_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        gray_im = cv2.resize(gray_im, (128, 128))
        #gray_im = cv2.bilateralFilter(gray_im,9,20,20)
        cv2.imwrite(infilename, gray_im)

paths = [
    'data\\train',
    'data\\test'
]
dirname = os.path.dirname(__file__)
for index, path in enumerate(paths):
    paths[index] = os.path.join(dirname, path)

for path in paths:
    for filedir in find_dirs(path, '*'):
        dir_path = os.path.join(path, filedir)
        for filename in find_files(dir_path, '*'):
            infilename = os.path.join(filedir, filename)
            if not os.path.isfile(infilename): continue
            prepering_data(infilename)

