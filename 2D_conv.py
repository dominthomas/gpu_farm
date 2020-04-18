import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import add
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.utils import multi_gpu_model
import os
import re
import gc
import random
import cv2
import matplotlib.image as mpimg

K.clear_session()
physical_devices = tf.config.experimental.list_physical_devices('GPU')
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
config = tf.config.experimental.set_memory_growth(physical_devices[1], True)
config = tf.config.experimental.set_memory_growth(physical_devices[2], True)
config = tf.config.experimental.set_memory_growth(physical_devices[3], True)
config = tf.config.experimental.set_memory_growth(physical_devices[4], True)

ad_files = os.listdir("/home/k1651915/OASIS/2D/AD/")
cn_files = os.listdir("/home/k1651915/OASIS/2D/CN/")

sub_id_ad = []
sub_id_cn = []
for file in ad_files:
    sub_id = re.search('(OAS\\d*)', file).group(1)
    if sub_id not in sub_id_ad:
        sub_id_ad.append(sub_id)

for file in cn_files:
    sub_id = re.search('(OAS\\d*)', file).group(1)
    if sub_id not in sub_id_cn:
        sub_id_cn.append(sub_id)


def crop(img, tol=0):
    # img is 2D image data
    # tol  is tolerance
    mask = img > tol
    m, n = img.shape
    mask0, mask1 = mask.any(0), mask.any(1)
    col_start, col_end = mask0.argmax(), n - mask0[::-1].argmax()
    row_start, row_end = mask1.argmax(), m - mask1[::-1].argmax()
    return img[row_start:row_end, col_start:col_end]


def get_images(folders, train=False, same_length=False, data_length=0, adni=False):

    return_list = []
    for folder in folders:

        if same_length and len(return_list) == data_length:
            return return_list

        file_num_only = []
        os.chdir(folder)
        files = os.listdir('.')

        for png_file in files:
            file_num_only.append(int(re.search('(\\d*)', png_file).group(1)))

        file_num_only.sort()
        png0 = mpimg.imread(str(file_num_only[86]) + ".png")
        png1 = mpimg.imread(str(file_num_only[87]) + ".png")
        png2 = mpimg.imread(str(file_num_only[88]) + ".png")

        png0 = png0[:, :, 1]
        png1 = png1[:, :, 1]
        png2 = png2[:, :, 1]

        if adni:
            png0 = get_rotated_images(png0, custom_angle=True, angle=-180)[0][:, :, 0]
            png1 = get_rotated_images(png1, custom_angle=True, angle=-180)[0][:, :, 0]
            png2 = get_rotated_images(png2, custom_angle=True, angle=-180)[0][:, :, 0]

        if train:
            return_list = return_list + get_rotated_images(png0)
            return_list = return_list + get_rotated_images(png1)
            return_list = return_list + get_rotated_images(png2)

        png0 = crop(png0)
        png1 = crop(png1)
        png2 = crop(png2)

        png0 = cv2.resize(png0, (227, 227))
        png1 = cv2.resize(png1, (227, 227))
        png2 = cv2.resize(png2, (227, 227))

        png0 = np.stack((png0,) * 1, axis=2)
        png1 = np.stack((png1,) * 1, axis=2)
        png2 = np.stack((png2,) * 1, axis=2)

        return_list = return_list + [png0, png1, png2]
        os.chdir('../')

    return return_list


def get_rotated_images(png, custom_angle=False, angle=0):
    angles = [3, -4, -3, 4, 1, 2, -1, -2]
    rotated_pngs = []

    if custom_angle:
        angles = [angle]
    else:
        png = get_rotated_images(png, True, -90)[0][:, :, 0]

    (h, w) = png.shape[:2]
    center = (w / 2, h / 2)

    for angle in angles:
        m = cv2.getRotationMatrix2D(center, angle, 1.0)
        r = cv2.warpAffine(png, m, (h, w))
        r = cv2.resize(crop(r), (227, 227))
        r = np.stack((r,) * 1, axis=2)
        rotated_pngs.append(r)

    return rotated_pngs


results = []
seeds = range(1, 500)

for i in seeds:
    random.seed(i)
    random.shuffle(sub_id_ad)
    random.shuffle(sub_id_cn)

    os.chdir("/home/k1651915/OASIS/2D/AD/")
    ad_sub_train = sub_id_ad[0:164]
    ad_sub_validate = sub_id_ad[165:169]
    ad_sub_test = sub_id_ad[170:177]

    ad_sub_train_files = []
    ad_sub_validate_files = []
    ad_sub_test_files = []

    for file in ad_files:
        file_sub_id = re.search('(OAS\\d*)', file).group(1)
        if file_sub_id in ad_sub_train:
            ad_sub_train_files.append(file)
        elif file_sub_id in ad_sub_validate:
            ad_sub_validate_files.append(file)
        elif file_sub_id in ad_sub_test:
            ad_sub_test_files.append(file)

    os.chdir("/home/k1651915/OASIS/2D/CN")

    cn_sub_train = sub_id_cn[0:573]
    cn_sub_validate = sub_id_cn[574:579]
    cn_sub_test = sub_id_cn[580:587]

    cn_sub_train_files = []
    cn_sub_validate_files = []
    cn_sub_test_files = []

    for file in cn_files:
        file_sub_id = re.search('(OAS\\d*)', file).group(1)
        if file_sub_id in cn_sub_train:
            cn_sub_train_files.append(file)
        elif file_sub_id in cn_sub_validate:
            cn_sub_validate_files.append(file)
        elif file_sub_id in cn_sub_test:
            cn_sub_test_files.append(file)

    os.chdir('/home/k1651915/OASIS/2D/AD/')
    ad_train = get_images(ad_sub_train_files, True)
    ad_validate = get_images(ad_sub_validate_files, same_length=True, data_length=5)
    ad_test = get_images(ad_sub_test_files, same_length=True, data_length=8)

    os.chdir('/home/k1651915/OASIS/2D/CN/')
    cn_train = get_images(cn_sub_train_files, True)
    cn_validate = get_images(cn_sub_validate_files, same_length=True, data_length=5)
    cn_test = get_images(cn_sub_test_files, same_length=True, data_length=8)
    print(len(ad_train))
    print(len(cn_train))
    print(len(ad_validate))
    print(len(cn_validate))
    print(len(ad_test))
    print(len(cn_test))

    train = np.asarray(cn_train + ad_train)
    validate = np.asarray(cn_validate + ad_validate)
    test = np.asarray(cn_test + ad_test)

    y1 = np.zeros(len(cn_train))
    y2 = np.ones(len(ad_train))
    train_labels = np.concatenate((y1, y2), axis=None)

    y1 = np.zeros(len(cn_validate))
    y2 = np.ones(len(ad_validate))
    validation_labels = np.concatenate((y1, y2), axis=None)

    y1 = np.zeros(len(cn_test))
    y2 = np.ones(len(ad_test))
    test_labels = np.concatenate((y1, y2), axis=None)

    cn_train = None
    ad_train = None
    cn_validate = None
    ad_validate = None
    cn_test = None
    ad_test = None
    gc.collect()

    #################################################

    with tf.device("/cpu:0"):
        with tf.device("/gpu:0"):
            model = tf.keras.Sequential()

            model.add(Conv2D(32,
                             input_shape=(227, 227, 1),
                             data_format='channels_last',
                             kernel_size=(7, 7),
                             strides=(4, 4),
                             padding='valid',
                             activation='relu'))

        with tf.device("/gpu:1"):
            model.add(MaxPooling2D(pool_size=(2, 2),
                                   strides=(2, 2),
                                   padding='valid'))

            model.add(Conv2D(64,
                             kernel_size=(5, 5),
                             strides=(1, 1),
                             padding='valid',
                             activation='relu'))

            model.add(MaxPooling2D(pool_size=(2, 2),
                                   strides=(2, 2),
                                   padding='valid'))

            model.add(Conv2D(384,
                             kernel_size=(3, 3),
                             strides=(1, 1),
                             padding='valid',
                             activation='relu'))

        with tf.device("/gpu:2"):
            model.add(Conv2D(384,
                             kernel_size=(3, 3),
                             strides=(1, 1),
                             padding='valid',
                             activation='relu'))

        with tf.device("/gpu:3"):
            model.add(Conv2D(512,
                             kernel_size=(3, 3),
                             strides=(1, 1),
                             padding='valid',
                             activation='relu'))

        with tf.device("/gpu:4"):
            model.add(Conv2D(256,
                             kernel_size=(3, 3),
                             strides=(1, 1),
                             padding='valid',
                             activation='relu'))

            model.add(MaxPooling2D(pool_size=(2, 2),
                                   strides=(2, 2),
                                   padding='valid'))

            model.add(Flatten())

            model.add(Dense(32, activation='relu'))

            model.add(Dense(1, activation='sigmoid'))
    # model = multi_gpu_model(model, gpus=5, cpu_merge=True, cpu_relocation=True)

    model.compile(loss=tf.keras.losses.binary_crossentropy,
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(train, train_labels,
              epochs=50,
              batch_size=512,
              shuffle=True,
              validation_data=(validate, validation_labels))

    #################################################

    train = None
    validate = None
    gc.collect()

    evaluation = model.evaluate(test, test_labels, verbose=0)

    test = None
    gc.collect()
    print(evaluation)
    results.append(evaluation[1])
    print("iteration:")
    print(i)
    print("mean:")
    print((sum(results) / len(results)))

    os.chdir('/home/k1651915/')
    with open('2D_conv.txt', 'a') as f:
        f.write("%s\n" % evaluation[1])

    K.clear_session()
    gc.collect()
