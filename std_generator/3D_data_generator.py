import numpy as np
import nibabel
import tensorflow as tf
from tensorflow.keras.layers import Conv3D, MaxPooling3D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras import backend as K
from data_helper import DataGenerator
import os
import gc
import random

"""@author Domin Thomas"""
"""Make sure that the working directory for this python script is in the '/home/k1651915/OASIS/3D/all/' , 
or in the ADNI 3D/all/ subdirectory depending on the training dataset """

"""Configure GPUs to prevent OOM errors"""
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

"""Retrieve file names"""
ad_files = os.listdir("/home/k1651915/OASIS/3D/ad/")
cn_files = os.listdir("/home/k1651915/OASIS/3D/cn/")

sub_id_ad = []
sub_id_cn = []

"""OASIS AD: 178 Subjects, 278 3T MRIs"""
"""OASIS CN: 588 Subjects, 1640 3T MRIs"""
"""Down-sampling CN to 278 MRIs"""
random.Random(129).shuffle(ad_files)
random.Random(129).shuffle(cn_files)
cn_files = cn_files[0:277]

"""Split files for training and validation"""
ad_train = ad_files[0:277]
cn_train = cn_files[0:277]

"""There is a chance subject bias could alter validation accuracy, but I don't care about this atm"""
"""TODO: In future, use samples from ADNI DataSet to account for Subject bias"""
# ad_validation = ad_files[276:277]
# cn_validation = cn_files[276:277]

"""Shuffle Train set"""
train = ad_train + cn_train
random.Random(129).shuffle(train)

"""Create a data dictionary with MRI file names"""
partition = {'train': train}
labels = {}
for item in ad_files:
    labels[item] = 1
for item in cn_files:
    labels[item] = 0

########################################################################################

with tf.device("/cpu:0"):
    with tf.device("/gpu:0"):
        model = tf.keras.Sequential()

        model.add(Conv3D(64,
                         input_shape=(100, 100, 100, 1),
                         data_format='channels_last',
                         kernel_size=(7, 7, 7),
                         strides=(2, 2, 2),
                         padding='valid',
                         activation='relu'))

    with tf.device("/gpu:1"):
        model.add(Conv3D(64,
                         kernel_size=(3, 3, 3),
                         padding='valid',
                         activation='relu'))

    with tf.device("/gpu:2"):
        model.add(Conv3D(128,
                         kernel_size=(3, 3, 3),
                         padding='valid',
                         activation='relu'))

        model.add(MaxPooling3D(pool_size=(2, 2, 2),
                               padding='valid'))

    with tf.device("/gpu:3"):
        model.add(Conv3D(128,
                         kernel_size=(3, 3, 3),
                         padding='valid',
                         activation='relu'))

        model.add(MaxPooling3D(pool_size=(2, 2, 2),
                               padding='valid'))

    with tf.device("/gpu:4"):
        model.add(Conv3D(128,
                         kernel_size=(3, 3, 3),
                         padding='valid',
                         activation='relu'))

        model.add(MaxPooling3D(pool_size=(2, 2, 2),
                               padding='valid'))

        model.add(Flatten())

        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.7))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.7))
        model.add(Dense(2, activation='softmax'))

model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adagrad(0.01),
              metrics=['accuracy'])

########################################################################################

"""Change working directory to OASIS/3D/all/"""
os.chdir("/home/k1651915/OASIS/3D/all/")
params = {'dim': (100, 100, 100),
          'batch_size': 6,
          'n_classes': 2,
          'n_channels': 1,
          'shuffle': True}
training_generator = DataGenerator(partition['train'], labels, **params)
model.fit_generator(generator=training_generator,
                    epochs=60)

"""Load test data from ADNI, 50 AD & 50 CN MRIs"""
ad_test_files = os.listdir("/home/k1651915/ADNI/3D/resized_ad/")
cn_test_files = os.listdir("/home/k1651915/ADNI/3D/resized_cn/")
random.Random(921).shuffle(ad_test_files)
random.Random(921).shuffle(cn_test_files)
ad_test_files = ad_test_files[0:50]
cn_test_files = cn_test_files[0:50]

"""Function to load 3D-MRI voxels"""


def get_images(files):
    return_list = []
    for file in files:
        nifti_data = np.asarray(nibabel.load(file).get_fdata())
        xs, ys, zs = np.where(nifti_data != 0)
        nifti_data = nifti_data[min(xs):max(xs) + 1, min(ys):max(ys) + 1, min(zs):max(zs) + 1]
        nifti_data = nifti_data[0:100, 0:100, 0:100]
        nifti_data = np.reshape(nifti_data, (100, 100, 100, 1))
        return_list.append(nifti_data)
    return return_list


os.chdir("/home/k1651915/ADNI/3D/resized_ad/")
ad_test = np.asarray(get_images(ad_test_files))
os.chdir("/home/k1651915/ADNI/3D/resized_cn/")
cn_test = np.asarray(get_images(cn_test_files))

ad_test_labels = tf.keras.utils.to_categorical(np.ones(50), 2)
cn_test_labels = tf.keras.utils.to_categorical(np.zeros(50), 2)

evaluation_ad = model.evaluate(ad_test, ad_test_labels, verbose=0)
evaluation_cn = model.evaluate(cn_test, cn_test_labels, verbose=0)

print("AD: ", evaluation_ad[1])
print("CN: ", evaluation_cn[1])

with open("/home/k1651915/3.txt", "a") as f:
    f.write("%s\n" % evaluation_ad[1])
    f.write("%s\n" % evaluation_cn[1])
    f.write("%s\n" % "========")

K.clear_session()
gc.collect()
