from PIL import Image
import os
from os.path import isfile, join

def conver_all_jpgs_to_pngs(path):
    print(path)
    for image_name in os.listdir(path):
        # print(image_name)
        filename, file_ext = os.path.splitext(image_name)
        if file_ext == '.jpg':
            print(filename)
            im = Image.open(os.path.abspath(path) + "\\" + image_name)
            im.save(os.path.abspath(path) + "\\" + filename + '.png')

if __name__ == '__main__':
    paths = [
        os.path.normpath("data/train/normal"),
        os.path.normpath("data/train/pneumonia"),
        os.path.normpath("data/train/COVID-19"),
        os.path.normpath("data/test/normal"),
        os.path.normpath("data/test/pneumonia"),
        os.path.normpath("data/test/COVID-19"),
    ]
    for path in paths:
        conver_all_jpgs_to_pngs(path)