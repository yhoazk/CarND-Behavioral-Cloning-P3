

import cv2
from PIL import Image
import matplotlib.pyplot as plt
import random
import numpy as np

import matplotlib
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

IMG = "D:/workspace/simulator-linux/center_2016_12_01_13_31_12_937.jpg"
LBL = -0.55

def shift_lr_random(image, label =None):
    shift = random.randint(-10, 10)
    shift_lbl = (0.005 * shift) + label
    M = np.float32([[1, 0, shift], [0, 1, 0]])
    return (cv2.warpAffine(image, M, (image.shape[1], image.shape[0])), shift_lbl)


def shift_ul_random(image, label):
    shift = random.randint(-10, 10)
    shift_lbl = (0.005 * shift) + label
    M = np.float32([[1, 0, 0], [0, 1, shift]])
    return (cv2.warpAffine(image, M, (image.shape[1], image.shape[0])), shift_lbl)

def mirror(image, label):
    mirr_img = cv2.flip(image,1)
    label = label * -1. ## add little offset so the image is not exactly the same
    return (np.add(mirr_img, random.uniform (-0.5, 0.5) ), label)

def add_squares(image, label):
    n = random.randint(3, 15)
    x = random.randint(10, 320 - 10)
    y = random.randint(10, 90 - 10)
    size = random.randint(5, 20)
    patches = []
    for i in range(N):
        polygon = Rectangle((x,y), size, size)
        patches.append(polygon)

    p = PatchCollection(patches)
    #p.set_array(np.zeros())



def plot_imgArr(img_arr, label=None, predict=None, gray=False):
    f, arr = plt.subplots(5,3)
    print(img_arr[0].shape)
    for n, subplt in enumerate(arr.reshape(-1)):
        if gray:
            subplt.imshow(img_arr[n],  cmap='gray')
        else:
            subplt.imshow(img_arr[n])
        subplt.axis('off')
        if label is not None and predict is None:
            subplt.set_title("st: "+str(label[n]))
        elif label is not None and predict is not None:
            subplt.set_title("st:"+str(label[n]) + "p:"+str(predict[n]))
    plt.show()

def preprocessrgb2gray(image_path):
    return np.multiply(np.mean(np.asarray(Image.open(image_path).crop((0, 50, 320, 140))), -1), 1 / 255)

if __name__ == "__main__":
    img =preprocessrgb2gray(IMG)

    #print(img[0][0:45])
    im,lbl = shift_lr_random(img, LBL)
    imgs = []
    lbls = []
    for n in range(5):
        im,lb = shift_lr_random(img, LBL)
        imgs.append(im)
        lbls.append(lb)

    for n in range(5):
        im,lb = mirror(img, LBL)
        imgs.append(im)
        lbls.append(lb)

    for n in range(5):
        im, lb = shift_ul_random(img, LBL)
        imgs.append(im)
        lbls.append(lb)

    plot_imgArr(img_arr=imgs, label=lbls, gray=True)
    plt.show()