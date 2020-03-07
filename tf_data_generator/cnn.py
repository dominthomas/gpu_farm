import numpy as np
import nibabel
import tensorflow as tf
from tensorflow.keras.layers import Conv3D, MaxPooling3D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras import Model
import tempfile
import random
import sys
import os
import gc

"""@author Domin Thomas"""
"""Make sure that the working directory for this python script is in the '/home/k1651915/OASIS/3D/all/' , 
or in the ADNI 3D/all/ subdirectory depending on the training dataset """

# tf.compat.v1.reset_default_graph()
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())

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

"""Split files for training"""
ad_train = ad_files[0:276]
cn_train = cn_files[0:276]

"""Shuffle Train data and Train labels"""
train = ad_train + cn_train
labels = np.concatenate((np.ones(len(ad_train)), np.zeros(len(cn_train))), axis=None)
random.Random(129).shuffle(train)
random.Random(129).shuffle(labels)
print(len(train))
print(len(labels))

"""Change working directory to OASIS/3D/all/"""
os.chdir("/home/k1651915/OASIS/3D/all/")

"""Create tf data pipeline"""


def load_image(file, label):
    nifti = np.asarray(nibabel.load(file.numpy().decode('utf-8')).get_fdata()).astype(np.float32)

    xs, ys, zs = np.where(nifti != 0)
    nifti = nifti[min(xs):max(xs) + 1, min(ys):max(ys) + 1, min(zs):max(zs) + 1]
    nifti = nifti[0:100, 0:100, 0:100]
    nifti = np.reshape(nifti, (100, 100, 100, 1))
    # nifti = tf.reshape(nifti, [1, 100, 100, 100, 1])
    return nifti, label


@tf.autograph.experimental.do_not_convert
def load_image_wrapper(file, label):
    return tf.py_function(load_image, [file, label], [tf.float64, tf.float64])


dataset = tf.data.Dataset.from_tensor_slices((train, labels))
dataset = dataset.map(load_image_wrapper, num_parallel_calls=12)
dataset = dataset.batch(12, drop_remainder=True).repeat()
print(tf.compat.v1.data.get_output_shapes(dataset))
dataset = dataset.unbatch()
print(tf.compat.v1.data.get_output_shapes(dataset))
dataset = dataset.prefetch(buffer_size=2)
iterator = iter(dataset)


def get_batch():
    image_batch, label_batch = iterator.get_next()

    i1 = image_batch[0, :, :, :, :]
    i2 = image_batch[1, :, :, :, :]
    i3 = image_batch[2, :, :, :, :]
    i4 = image_batch[3, :, :, :, :]
    i5 = image_batch[4, :, :, :, :]
    i6 = image_batch[5, :, :, :, :]
    i7 = image_batch[6, :, :, :, :]
    i8 = image_batch[7, :, :, :, :]
    i9 = image_batch[8, :, :, :, :]
    i10 = image_batch[9, :, :, :, :]
    i11 = image_batch[10, :, :, :, :]
    i12 = image_batch[11, :, :, :, :]

    image_batch = tf.stack([i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12])
    return [image_batch, label_batch]


batch = get_batch()
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
        model.add(Dense(1, activation='sigmoid'))

model.compile(loss=tf.keras.losses.binary_crossentropy,
              optimizer=tf.keras.optimizers.Adagrad(0.01),
              metrics=['accuracy'])
########################################################################################
model.fit(batch, steps_per_epoch=46, epochs=50)
########################################################################################

"""Load test data from ADNI, 50 AD & 50 CN MRIs"""
test_size = 5
ad_test_files = os.listdir("/home/k1651915/ADNI/3D/resized_ad/")
cn_test_files = os.listdir("/home/k1651915/ADNI/3D/resized_cn/")
random.Random(921).shuffle(ad_test_files)
random.Random(921).shuffle(cn_test_files)
ad_test_files = ad_test_files[0:test_size]
cn_test_files = cn_test_files[0:test_size]

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

ad_test_labels = tf.keras.utils.to_categorical(np.ones(test_size), 2)
cn_test_labels = tf.keras.utils.to_categorical(np.zeros(test_size), 2)

# ad_test_labels = np.ones(test_size), 2
# cn_test_labels = np.zeros(test_size), 2

evaluation_ad = model.evaluate(ad_test, ad_test_labels, verbose=0)
evaluation_cn = model.evaluate(cn_test, cn_test_labels, verbose=0)

print("AD: ", evaluation_ad[1])
print("CN: ", evaluation_cn[1])

with open("/home/k1651915/3.txt", "a") as f:
    f.write("%s\n" % evaluation_ad[1])
    f.write("%s\n" % evaluation_cn[1])
    f.write("%s\n" % "========")

# K.clear_session()
gc.collect()
