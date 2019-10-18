import os
import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt


def readTrafficSigns(rootpath):
    '''Reads traffic sign data for German Traffic Sign Recognition Benchmark.

    Arguments: path to the traffic sign data, for example './GTSRB/Training'
    Returns:   list of images, list of corresponding labels'''
    images = [] # images
    labels = [] # corresponding labels
    # loop over all 42 classes
    for c in range(0,43):
        prefix = rootpath + '/' + format(c, '05d') + '/' # subdirectory for class
        gtFile = open(prefix + 'GT-'+ format(c, '05d') + '.csv') # annotations file
        gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
        next(gtReader) # skip header
        # loop over all images in current annotations file
        for row in gtReader:
            images.append(plt.imread(prefix + row[0])) # the 1th column is the filename
            labels.append(row[7]) # the 8th column is the label
        gtFile.close()
    return images, labels


def reshape_img(images):
    result = list()
    for img in images:
        height, width = img.shape[1], img.shape[0]
        if width > height:
            left = np.ceil((width - height) / 2).astype(int)
            right = np.floor((width - height) / 2).astype(int)
            top, bottom = 0, 0
        elif height > width:
            bottom = np.ceil((height - width) / 2).astype(int)
            top = np.floor((height - width) / 2).astype(int)
            left, right = 0, 0
        img = np.lib.pad(img, ((top, bottom), (left, right)), cv2.BORDER_REPLICATE)
        img = cv2.resize()
        result.append(img)
    # plt.imshow(result[0])
    # plt.show()
    # print(result[0].shape)
    return result


# trainImages, trainLabels = readTrafficSigns('GTSRB/Final_Training/Images')
# print(len(trainLabels), len(trainImages))
# print(trainLabels[39208])
# print(trainImages[39208].shape)
# plt.imshow(trainImages[39208])
# plt.show()
print(plt.imread('test.jpg').shape)
reshape_img((plt.imread('test.jpg'), ))

