import cv2
import csv
import random
import time as lame
import numpy as np
import collections
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage import transform
from skimage.util import random_noise
from skimage import exposure
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import model_selection


resize = (30, 30)
CLASSES_FREQUENCY = 0

def readTrafficSigns(rootpath):
    '''Reads traffic sign data for German Traffic Sign Recognition Benchmark.

    Arguments: path to the traffic sign data, for example './GTSRB/Training'
    Returns:   list of images, list of corresponding labels'''
    images = [] # images
    labels = [] # corresponding labels
    print('\n -- Reading data Train Data-- ')
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


def readTest(rootpath):
    images = [] # images
    labels = [] # corresponding labels
    print('\n -- Reading data Testing Data-- ')

    prefix = rootpath + '/'
    gtFile = open(prefix + 'GT-final_test' + '.csv') # annotations file
    gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
    next(gtReader) # skip header
    # loop over all images in current annotations file
    for row in gtReader:
        images.append(plt.imread(prefix + row[0])) # the 1th column is the filename
        labels.append(row[7]) # the 8th column is the label
    gtFile.close()
    print(len(images))
    return images, labels


def reshape_img(images, shape=resize):
    result = list()

    # Loop over all images
    best = 0
    best_img = [0, 0]
    flag = False
    for img in images:
        # Get values for either horizontal or vertical padding
        height, width = img.shape[1], img.shape[0]
        top, bottom, left, right = 0, 0, 0, 0

        # Get the most unsquare image
        if height // width > best or width // height > best:
            best = height // width if height // width > best else width // height
            best_img[0] = img
            flag = True

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

        # Get the most unsquare image
        if flag:
            best_img[1] = img
            flag = False

    # plt.imshow(best_img[0])
    # plt.title('Before Transform')
    # plt.show()
    # plt.imshow(best_img[1])
    # plt.title('After Transform')
    # plt.show()
    # print(best_img[0].shape)
    return result


def data_split(images, labels, augment=True):
    print('\n -- Splitting the data -- ')

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
    print('\n -- Data Splitting -- ')
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

    # getting info of classes for frequency
    counter = {int(k):int(v) for k,v in collections.Counter(train_labels).items()}

    # plt.bar(list(counter.keys()), list(counter.values()))
    # plt.title('Train Split Class Frequencies Before Augmentation')
    # plt.show()

    # Augmentation
    if augment:
        print('\n -- Augmentation --')
        train_set, train_labels = augmentation(train_set, train_labels, counter)

    # Combine images and labels so we can shuffle them safely
    train = list(zip(train_set, train_labels))
    test = list(zip(test_set, test_labels))

    # Shuffle the data
    np.random.shuffle(train)
    np.random.shuffle(test)

    train_set, train_labels = list(zip(*train))
    test_set, test_labels = list(zip(*test))

    return train_set, train_labels, test_set, test_labels


def augmentation(images, labels, class_freq):
    max_class_size = np.max(list(class_freq.values()))
    for i in tqdm(range(0, 43)):  # Go through all the classes
        number_of_images = class_freq[i]
        start = sum(list(class_freq.values())[:i])
        end = start + number_of_images
        class_images = images[start:end]
        while number_of_images < max_class_size:  # If class has less images than it should
            for image in class_images:  # Go through each image until we are done
                images.append(single_image_augmentation(image))
                labels.append(i)

                number_of_images += 1  # Check # of images
                if number_of_images == max_class_size:
                    break
    return images, labels


def single_image_augmentation(image):
    # plt.imshow(image)
    # plt.title('Initial Image')
    # plt.show()

    transformed_image = transform.rotate(image, random.uniform(-20, 20))  # Rotate
    noised_image = random_noise(transformed_image)  # ('s&p', clip=True, amount=random.uniform(0, 0.065))
    gamma_corrected_image = exposure.adjust_gamma(noised_image, gamma=random.uniform(0, 2))  # Brightness

    # plt.imshow(gamma_corrected_image)
    # plt.title('Augmented Image')
    # plt.show()
    return gamma_corrected_image


def normalize(images):
    for i in range(len(images)):
        images[i] = images[i].flatten()
        images[i] = (images[i] / 255.0)
    return images


def experimentation(trainImages, trainLabels):
    """No augmentation"""
    X_train, y_train, X_test, y_test = data_split(trainImages, trainLabels, augment=False)
    print('\n -- Image Transformation -- ')
    X_train_new = reshape_img(X_train)
    X_test_new = reshape_img(X_test)
    print('\n -- Image Normalization --')
    X_train_new = normalize(X_train_new)
    X_test_new = normalize(X_test_new)

    classifier = RandomForestClassifier(n_jobs=-1, n_estimators=60)
    print('\n -- Training -- ')
    classifier.fit(X_train_new, y_train)
    print('\n -- Evaluating -- ')
    y_pred = classifier.predict(X_test_new)
    print('\n ************** Results WITHOUT Augmentation **************')
    # print(classification_report(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))

    """Augmentation is used"""
    X_train, y_train, X_test, y_test = data_split(trainImages, trainLabels)
    print('\n -- Image Transformation -- ')
    X_train_new = reshape_img(X_train)
    X_test_new = reshape_img(X_test)
    print('\n -- Image Normalization --')
    X_train_new = normalize(X_train_new)
    X_test_new = normalize(X_test_new)

    classifier = RandomForestClassifier(n_jobs=-1, n_estimators=60)
    print('\n -- Training -- ')
    classifier.fit(X_train_new, y_train)
    print('\n -- Evaluating -- ')
    y_pred = classifier.predict(X_test_new)
    print('\n ************** Results WITH Augmentation **************')
    # print(classification_report(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))

    """Different Image Sizes"""
    times = list()
    accuracies = list()

    # Images from 20 to 60
    shapes = [(20 + i * 10, 20 + i * 10) for i in range(5)]

    X_train, y_train, X_test, y_test = data_split(trainImages, trainLabels)

    for new_shape in shapes:
        start_time = lame.time()
        classifier = RandomForestClassifier(n_jobs=-1, n_estimators=60)

        print('\n -- Image Transformation -- ')
        X_train_new = reshape_img(X_train, new_shape)
        X_test_new = reshape_img(X_test, new_shape)

        print('\n -- Image Normalization --')
        X_train_new = normalize(X_train_new)
        X_test_new = normalize(X_test_new)

        print('\n -- Training -- ')
        classifier.fit(X_train_new, y_train)
        print('\n -- Evaluating -- ')
        y_pred = classifier.predict(X_test_new)
        print('\n ************** Results WITH IMG SIZE: ', new_shape, ' **************')
        # print(classification_report(y_test, y_pred))
        accuracy = accuracy_score(y_test, y_pred)
        end_time = lame.time()

        # Get time and accuracy of each size
        time = end_time - start_time
        times.append(end_time - start_time)
        accuracies.append(accuracy)

        print('Time: ', time, '\nAccuracy: ', accuracy)

    plt.plot(shapes, times)
    plt.xlabel('shape')
    plt.ylabel('time')
    plt.title('Shape-Time Dependence')
    plt.show()

    plt.plot(shapes, accuracies)
    plt.xlabel('shape')
    plt.ylabel('accuracy')
    plt.title('Shape-Accuracy Dependence')
    plt.show()


def parameter_tuning(trainImages, trainLabels):
    X_train, y_train, X_test, y_test = data_split(trainImages, trainLabels)
    print('\n -- Image Transformation -- ')
    X_train = reshape_img(X_train)
    X_test = reshape_img(X_test)

    print('\n -- Image Normalization --')
    X_train = normalize(X_train)
    X_test = normalize(X_test)
    best_accuracy = -1
    best_parameter = 0
    results = list()

    for i in range(128, 512, 64):  # Test for different parameters from 128 to 512 with step = 64
        classifier = RandomForestClassifier(n_jobs=-1, n_estimators=i)
        print('\n -- Training -- ')
        classifier.fit(X_train, y_train)

        print('\n -- Evaluating -- ')
        y_pred = classifier.predict(X_test)
        result = accuracy_score(y_test, y_pred)
        if result > best_accuracy:
            best_accuracy = result
            best_parameter = i
            results.append((result, i))
        print('\n ************** Results with n_estimators = ', i, ' **************')
        # print(classification_report(y_test, y_pred))
        print('Current Accuracy: ', result)
    print('\n Best result was *', best_accuracy, '* with n_estimateor = ', best_parameter)
    print(results)


def start_model(trainImages, trainLabels, testImages, testLabels, n_estimators=60, depth=None):
    X_train, y_train, X_val, y_val = data_split(trainImages, trainLabels)

    print('\n -- Image Transformation -- ')
    X_train = reshape_img(X_train)
    testImages = reshape_img(testImages)

    print('\n -- Image Normalization --')
    X_train = normalize(X_train)
    testImages = normalize(testImages)

    print('\n -- Training -- ')
    classifier = RandomForestClassifier(n_jobs=-1, n_estimators=n_estimators, max_depth=depth)
    classifier.fit(X_train, y_train)

    print('\n -- Evaluating -- ')
    y_pred = classifier.predict(testImages)
    # print(confusion_matrix(y_val, y_pred))
    print(classification_report(testLabels, y_pred))
    print(accuracy_score(testLabels, y_pred))


def main():

    """Use for experiments only:"""
    # experimentation(trainImages, trainLabels)
    # parameter_tuning(trainImages, trainLabels)

    while True:
        print('\nPrint \'exit\' to stop')
        while True:
            n_estimators = input('Enter number of estimators (if 0 then default=60): ')
            if n_estimators == 'exit':
                return
            try:
                n_estimators = int(n_estimators)
            except ValueError:
                print('Enter valid positive number or exit')
                continue

            if n_estimators < 0:
                print('Enter valid positive number or exit')
            elif n_estimators == 0:
                n_estimators = 60
                break
            else:
                break

        while True:
            depth = input('Enter the depth (if 0 then default=None): ')
            if depth == 'exit':
                return
            try:
                depth = int(depth)
            except ValueError:
                print('Enter valid positive number or exit')
                continue

            if depth < 0:
                print('Enter valid positive number or exit')
            elif depth == 0:
                depth=None
                break
            else:
                break

        trainImages, trainLabels = readTrafficSigns('GTSRB/Final_Training/Images')
        testImages, testLabels = readTest('GTSRB/Final_Test/Images')
        start_model(trainImages, trainLabels, testImages, testLabels, n_estimators, depth)


if __name__ == '__main__':
    main()