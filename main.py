import cv2
import csv
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import groupby
from skimage import transform
from skimage.util import random_noise
from skimage import exposure
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import model_selection
from sklearn.model_selection import StratifiedKFold



resize = (30, 30)
CLASSES_FREQUENCY = 0

def readTrafficSigns(rootpath):
    '''Reads traffic sign data for German Traffic Sign Recognition Benchmark.

    Arguments: path to the traffic sign data, for example './GTSRB/Training'
    Returns:   list of images, list of corresponding labels'''
    images = [] # images
    labels = [] # corresponding labels
    print('\n -- Reading data -- ')
    # loop over all 42 classes
    for c in tqdm(range(0,43)):
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


def reshape_img(images, shape):
    print('\n -- Image Reshaping --')
    result = list()

    # Loop over all images
    for img in images:
        # Get values for either horizontal or vertical padding
        height, width = img.shape[1], img.shape[0]
        top, bottom, left, right = 0, 0, 0, 0
        if width > height:  # get params for padding
            left = np.ceil((width - height) / 2).astype(int)
            right = np.floor((width - height) / 2).astype(int)
            top, bottom = 0, 0
        elif height > width:
            bottom = np.ceil((height - width) / 2).astype(int)
            top = np.floor((height - width) / 2).astype(int)
            left, right = 0, 0
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REPLICATE)  # add padding
        img = cv2.resize(img, shape)  # resize according to specifications
        result.append(img)
    # plt.imshow(result[0])
    # plt.show()
    # print(result[0].shape)
    return result


def data_split(images, labels, augment, shape = resize):

    print('\n -- Image Transformation -- ')
    images = reshape_img(images, shape)

    print('\n -- Splitting the data -- ')
    # getting info of classes for frequency
    existing_classes = [len(list(group)) for key, group in groupby(labels)]
    # print(existing_classes, np.max(existing_classes))
    # print(len(images))

    counter = 0
    random_split = random.random()

    train_set = list()
    train_labels = list()
    test_set = list()
    test_labels = list()

    """
    Loop over all images
    With random probability `random_split` the track goes either to training set or to the validation set
    """
    for i in tqdm(range(len(images))):
        if counter == 30:  # after each 30th pic new track begins
            counter = 0
            random_split = random.random()

        if random_split < .8:  # 80% go to training set
            train_set.append(images[i])
            train_labels.append(labels[i])
        else:  # 20% go to validation set
            test_set.append(images[i])
            test_labels.append(labels[i])
        counter += 1

    # Augmentation
    if augment:
        print('\n -- Augmentation --')
        train_set, train_labels = augmentation(train_set, train_labels)

    # Normalization
    print('\n -- Image Normalization --')
    train_set = normalize(train_set)
    test_set = normalize(test_set)

    # Combine images and labels so we can shuffle them safely
    train = list(zip(train_set, train_labels))
    test = list(zip(test_set, test_labels))

    # Shuffle the data
    np.random.shuffle(train)
    np.random.shuffle(test)

    train_set, train_labels = list(zip(*train))
    test_set, test_labels = list(zip(*test))

    # print(len(train_set), len(test), ' -- ', len(train_set) / len(images))

    # y_pos = np.arange(len(existing_classes))
    # plt.bar(y_pos, [len(list(group)) for key, group in groupby(train_labels)])
    # plt.show()
    return train_set, train_labels, test_set, test_labels


def augmentation(images, labels):
    class_freq = [len(list(group)) for key, group in groupby(labels)]
    # print('\n > Before augmentation: ', len(images), ' --- ', np.max(class_freq))
    # print(class_freq)
    max_class_size = np.max(class_freq)
    start = 0
    for i in tqdm(range(0, 43)):
        number_of_images = class_freq[i]
        start = sum(class_freq[:i])
        end = start + number_of_images
        class_images = images[start:end]
        while number_of_images != max_class_size:
            for image in class_images:
                images.append(single_image_augmentation(image))
                labels.append(i)

                number_of_images += 1
                if number_of_images == max_class_size:
                    break
    # print('\n > After augmentation: ', len(images))
    # print([len(list(group)) for key, group in groupby(labels)])
    # print(len([len(list(group)) for key, group in groupby(labels)]))
    return images, labels


def single_image_augmentation(image):
    transformed_image = transform.rotate(image, random.uniform(-20, 20))
    noised_image = random_noise(transformed_image)  # ('s&p', clip=True, amount=random.uniform(0, 0.065))
    gamma_corrected_image = exposure.adjust_gamma(noised_image, gamma=random.uniform(0, 2))
    # log_corrected_image = exposure.adjust_log(image)
    # show_images(image, gamma_corrected_image, 'gamma')
    # plt.imshow(gamma_corrected_image)
    # plt.show()
    return gamma_corrected_image


def normalize(images):
    for i in range(len(images)):
        images[i] = images[i].flatten()
        images[i] = images[i] / 255.0
    return images


def experimentation(trainImages, trainLabels):
    # No augmentation
    X_train, y_train, X_test, y_test = data_split(trainImages, trainLabels, augment=False)
    classifier = RandomForestClassifier(n_estimators=60)  # n_estimators=100, max_depth=100
    print('\n -- Training -- ')
    classifier.fit(X_train, y_train)

    print('\n -- Evaluating -- ')
    y_pred = classifier.predict(X_test)
    print('\n ************** Results WITHOUT Augmentation **************')
    print(classification_report(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))


    # Augmentation is used
    X_train, y_train, X_test, y_test = data_split(trainImages, trainLabels, augment=True)
    classifier = RandomForestClassifier(n_estimators=60)  # n_estimators=100, max_depth=100
    print('\n -- Training -- ')
    classifier.fit(X_train, y_train)

    print('\n -- Evaluating -- ')
    y_pred = classifier.predict(X_test)
    print('\n ************** Results WITH Augmentation **************')
    print(classification_report(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))


    # Different Image Sizes
    for i in range(6):
        new_shape = (30 + i * 10, 30 + i * 10)
        X_train, y_train, X_test, y_test = data_split(trainImages, trainLabels, augment=True, shape=new_shape)
        classifier = RandomForestClassifier(n_estimators=60)  # n_estimators=100, max_depth=100
        print('\n -- Training -- ')
        classifier.fit(X_train, y_train)

        print('\n -- Evaluating -- ')
        y_pred = classifier.predict(X_test)
        print('\n ************** Results WITH IMG SIZE: ',new_shape ,' **************')
        print(classification_report(y_test, y_pred))
        print(accuracy_score(y_test, y_pred))


def parameter_tuning(trainImages, trainLabels):
    X_train, y_train, X_test, y_test = data_split(trainImages, trainLabels, augment=True)
    best_accuracy = 0
    best_parameter = 0
    for i in range(10, 200):
        classifier = RandomForestClassifier(n_estimators=i)  # n_estimators=100, max_depth=100
        print('\n -- Training -- ')
        classifier.fit(X_train, y_train)

        print('\n -- Evaluating -- ')
        y_pred = classifier.predict(X_test)
        result = accuracy_score(y_test, y_pred)
        if result > best_accuracy:
            best_accuracy = result
            best_parameter = i
        print('\n ************** Results with n_estimators = ',i ,' **************')
        print(classification_report(y_test, y_pred))
        print(result)
    print('\n Best result was *', best_accuracy, '* with n_estimateor = ', best_parameter)

def main():
    trainImages, trainLabels = readTrafficSigns('GTSRB/Final_Training/Images')
    """
    Use for experiments only!
    """
    experimentation(trainImages, trainLabels)

    # X_train, y_train, X_test, y_test = data_split(trainImages, trainLabels, augment=True)
    # classifier = RandomForestClassifier(n_estimators=60, max_depth=70)  # n_estimators=100, max_depth=100
    # print('\n -- Training -- ')
    # classifier.fit(X_train, y_train)
    #
    # print('\n -- Evaluating -- ')
    # y_pred = classifier.predict(X_test)
    # # print(confusion_matrix(y_test, y_pred))
    # print(classification_report(y_test, y_pred))
    # print(accuracy_score(y_test, y_pred))

    # result = classifier.score(X_test, y_test)
    # print("Accuracy: %.2f%%" % (result * 100.0))



if __name__ == '__main__':
    main()