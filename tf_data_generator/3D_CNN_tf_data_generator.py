import numpy as np
import nibabel
import tensorflow as tf
from tensorflow.keras.layers import Conv3D, MaxPooling3D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras import Model
from keras import backend as K
import tempfile
import random
import sys
import os
import gc

"""@author Domin Thomas"""
"""Make sure that the working directory for this python script is in the '/home/k1651915/OASIS/3D/all/' , 
or in the ADNI 3D/all/ subdirectory depending on the training dataset """

sess = tf.Session()
K.set_session(sess)

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
train_labels = np.concatenate((np.ones(len(ad_train)), np.zeros(len(cn_train))), axis=None)
random.Random(129).shuffle(train)
random.Random(129).shuffle(train_labels)
print(len(train))
print(len(train_labels))

"""Change working directory to OASIS/3D/all/"""
os.chdir("/home/k1651915/OASIS/3D/all/")

"""Create tf data pipeline"""


def load_image(file, label):
    nifti = np.asarray(nibabel.load(file.numpy().decode('utf-8')).get_fdata()).astype(np.float32)

    xs, ys, zs = np.where(nifti != 0)
    nifti = nifti[min(xs):max(xs) + 1, min(ys):max(ys) + 1, min(zs):max(zs) + 1]
    nifti = nifti[0:100, 0:100, 0:100]
    nifti = np.reshape(nifti, (100, 100, 100, 1))
    nifti = tf.convert_to_tensor(nifti, tf.float64)
    return nifti, tf.cast(label, tf.int32)


@tf.autograph.experimental.do_not_convert
def load_image_wrapper(file, train_label):
    return tf.py_function(load_image, [file, train_labels], [tf.float64, tf.int32])


dataset = tf.data.Dataset.from_tensor_slices((train, train_labels))
dataset = dataset.map(load_image_wrapper, num_parallel_calls=12)
dataset = dataset.batch(12, drop_remainder=True) # Removed .repeat()
dataset = dataset.prefetch(buffer_size=2)
iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
batch = iterator.get_next()


########################################################################################

# x = tf.identity(features, name="input_tensor")
# x = tf.reshape(x, [-1, 100, 100, 100, 1])
# x = tf.identity(x, name="input_tensor_after")


with tf.device("/cpu:0"):
    img = tf.keras.backend.placeholder(dtype=tf.float64, shape=(100, 100, 100, 1))
    with tf.device("/gpu:0"):
        x = Conv3D(64,
                   data_format='channels_last',
                   kernel_size=(7, 7, 7),
                   strides=(2, 2, 2),
                   padding='valid',
                   activation='relu')(img)

        with tf.device("/gpu:1"):
            x = Conv3D(64,
                       kernel_size=(3, 3, 3),
                       padding='valid',
                       activation='relu')(x)

        with tf.device("/gpu:2"):
            x = Conv3D(128,
                       kernel_size=(3, 3, 3),
                       padding='valid',
                       activation='relu')(x)

            x = MaxPooling3D(pool_size=(2, 2, 2),
                             padding='valid')(x)

        with tf.device("/gpu:3"):
            x = Conv3D(128,
                       kernel_size=(3, 3, 3),
                       padding='valid',
                       activation='relu')(x)

            x = MaxPooling3D(pool_size=(2, 2, 2),
                             padding='valid')(x)

        with tf.device("/gpu:4"):
            x = Conv3D(128,
                       kernel_size=(3, 3, 3),
                       padding='valid',
                       activation='relu')(x)

            x = MaxPooling3D(pool_size=(2, 2, 2),
                             padding='valid')(x)

            x = Flatten()(x)

            x = Dense(256, activation='relu')(x)
            x = Dropout(0.7)(x)
            x = Dense(256, activation='relu')(x)
            x = Dropout(0.7)(x)
            preds = Dense(2, activation='softmax')(x)


labels = tf.keras.backend.placeholder(tf.int32, shape=(6, None))

loss = tf.math.reduce_mean(tf.keras.losses.SparseCategoricalCrossentropy(labels, preds))
train_step = tf.compat.v1.train.AdagradOptimizer(0.001).minimize(loss)

sess.run(tf.global_variables_initializer())

########################################################################################
epochs = 5
with sess.as_default():
    for i in range(epochs):
        sess.run(iterator.initializer)

        try:
            while True:
                image_batch = sess.run(batch)
                train_step.run(image_batch[0], image_batch[1])

        except tf.errors.OutOfRangeError:
            print('End of epoch...')
########################################################################################


# model.fit(batch_images, batch_labels, steps_per_epoch=92, epochs=50)

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

evaluation_ad = preds.evaluate(ad_test, ad_test_labels, verbose=0)
evaluation_cn = preds.evaluate(cn_test, cn_test_labels, verbose=0)

print("AD: ", evaluation_ad[1])
print("CN: ", evaluation_cn[1])

with open("/home/k1651915/3.txt", "a") as f:
    f.write("%s\n" % evaluation_ad[1])
    f.write("%s\n" % evaluation_cn[1])
    f.write("%s\n" % "========")

# K.clear_session()
gc.collect()