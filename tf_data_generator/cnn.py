import numpy as np
import nibabel
import tensorflow as tf
from tensorflow.keras.layers import Conv3D, MaxPooling3D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
import random
import os
import gc

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

"""OASIS AD: 178 Subjects, 278 3T MRIs"""
"""OASIS CN: 588 Subjects, 1640 3T MRIs"""
"""Down-sampling CN to 278 MRIs"""
random.Random(129).shuffle(ad_files)
random.Random(129).shuffle(cn_files)

batch_size = 261

"""Split files for training"""
ad_train = ad_files[0:batch_size]
cn_train = cn_files[0:batch_size]

"""Shuffle Train data and Train labels"""
train = ad_train + cn_train
labels = np.concatenate((np.ones(len(ad_train)), np.zeros(len(cn_train))), axis=None)
random.Random(129).shuffle(train)
random.Random(129).shuffle(labels)
print(len(train))
print(len(labels))


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


"""Change working directory to OASIS/3D/all/"""
os.chdir("/home/k1651915/OASIS/3D/all/")

"""Get Validation Data"""
"""
v_ad = ad_files[0:5]
v_cn = cn_files[0:5]
v_labels = np.concatenate((np.ones(5), np.zeros(5)), axis=None)
v_ad = get_images(v_ad)
v_cn = get_images(v_cn)
v = np.asarray(v_ad + v_cn)
"""
"""Create tf data pipeline"""


def load_image(file, label):
    nifti = np.asarray(nibabel.load(file.numpy().decode('utf-8')).get_fdata()).astype(np.float32)

    xs, ys, zs = np.where(nifti != 0)
    nifti = nifti[min(xs):max(xs) + 1, min(ys):max(ys) + 1, min(zs):max(zs) + 1]
    nifti = nifti[0:100, 0:100, 0:100]
    nifti = np.reshape(nifti, (100, 100, 100, 1))
    return nifti, label


@tf.autograph.experimental.do_not_convert
def load_image_wrapper(file, label):
    return tf.py_function(load_image, [file, label], [tf.float64, tf.float64])


dataset = tf.data.Dataset.from_tensor_slices((train, labels))
dataset = dataset.map(load_image_wrapper, num_parallel_calls=32)
dataset = dataset.prefetch(buffer_size=1)
dataset = dataset.apply(tf.data.experimental.prefetch_to_device('/device:GPU:0', 1))
dataset = dataset.batch((batch_size*2)).repeat()

iterator = iter(dataset)

batch_image, batch_label = iterator.get_next()

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
model.fit(batch_image, batch_label, epochs=70, batch_size=12)
########################################################################################

"""Load test data from ADNI, 50 AD & 50 CN MRIs"""
"""
test_size = 50
ad_test_files = os.listdir("/home/k1651915/ADNI/3D/resized_ad/")
cn_test_files = os.listdir("/home/k1651915/ADNI/3D/resized_cn/")
random.Random(921).shuffle(ad_test_files)
random.Random(921).shuffle(cn_test_files)
ad_test_files = ad_test_files[0:test_size]
cn_test_files = cn_test_files[0:test_size]

os.chdir("/home/k1651915/ADNI/3D/resized_ad/")
ad_test = np.asarray(get_images(ad_test_files))
os.chdir("/home/k1651915/ADNI/3D/resized_cn/")
cn_test = np.asarray(get_images(cn_test_files))

# ad_test_labels = tf.keras.utils.to_categorical(np.ones(test_size), 2)
# cn_test_labels = tf.keras.utils.to_categorical(np.zeros(test_size), 2)

ad_test_labels = np.concatenate((np.ones((test_size-1)), 1), axis=None)
cn_test_labels = np.concatenate((np.zeros((test_size-1)), 1), axis=None)

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
"""
# K.clear_session()
gc.collect()
ad_test_files = ad_files[261:278]
cn_test_files = cn_files[261:278]
test_size = len(ad_test_files)

ad_test_labels = np.concatenate((np.ones((test_size-1)), 1), axis=None)
cn_test_labels = np.concatenate((np.zeros((test_size-1)), 1), axis=None)

ad_test = np.asarray(get_images(ad_test_files))
cn_test = np.asarray(get_images(ad_test_files))

evaluation_ad = model.evaluate(ad_test, ad_test_labels, verbose=0)
evaluation_cn = model.evaluate(cn_test, cn_test_labels, verbose=0)

print("AD: ", evaluation_ad[1])
print("CN: ", evaluation_cn[1])

with open("/home/k1651915/3.txt", "a") as f:
    f.write("%s\n" % evaluation_ad[1])
    f.write("%s\n" % evaluation_cn[1])
    f.write("%s\n" % "========")

