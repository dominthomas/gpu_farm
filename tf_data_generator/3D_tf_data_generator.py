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
    nifti = tf.convert_to_tensor(nifti, tf.float64)
    nifti = tf.cast(nifti, tf.float32)
    return nifti, tf.cast(label, tf.int32)


@tf.autograph.experimental.do_not_convert
def load_image_wrapper(file, labels):
    return tf.py_function(load_image, [file, labels], [tf.float64, tf.int32])


def train_input():
    dataset = tf.data.Dataset.from_tensor_slices((train, labels))
    dataset = dataset.map(load_image_wrapper, num_parallel_calls=12)
    dataset = dataset.batch(12, drop_remainder=True).repeat()
    dataset = dataset.prefetch(buffer_size=2)
    return dataset


########################################################################################
# input_shape = (100, 100, 100, 1),


class CNN_Model(Model):
    with tf.device("/cpu:0"):
        def __init__(self):
            super(CNN_Model, self).__init__()
            with tf.device("/gpu:0"):
                self.conv1 = Conv3D(64,
                                    data_format='channels_last',
                                    kernel_size=(7, 7, 7),
                                    strides=(2, 2, 2),
                                    padding='valid',
                                    activation='relu')

                with tf.device("/gpu:1"):
                    self.conv2 = Conv3D(64,
                                        kernel_size=(3, 3, 3),
                                        padding='valid',
                                        activation='relu')

                with tf.device("/gpu:2"):
                    self.conv3 = Conv3D(128,
                                        kernel_size=(3, 3, 3),
                                        padding='valid',
                                        activation='relu')

                    self.maxPool1 = MaxPooling3D(pool_size=(2, 2, 2),
                                                 padding='valid')

                with tf.device("/gpu:3"):
                    self.conv4 = Conv3D(128,
                                        kernel_size=(3, 3, 3),
                                        padding='valid',
                                        activation='relu')

                    self.maxPool2 = MaxPooling3D(pool_size=(2, 2, 2),
                                                 padding='valid')

                with tf.device("/gpu:4"):
                    self.conv5 = Conv3D(128,
                                        kernel_size=(3, 3, 3),
                                        padding='valid',
                                        activation='relu')

                    self.maxPool3 = MaxPooling3D(pool_size=(2, 2, 2),
                                                 padding='valid')

                    self.flatten = Flatten()

                    self.dense1 = Dense(256, activation='relu')
                    self.dropout1 = Dropout(0.7)
                    self.dense2 = Dense(256, activation='relu')
                    self.dropout2 = Dropout(0.7)
                    self.dense3 = Dense(2, activation='softmax')

        def cnn_model(self, features):
            x = tf.identity(features, name="input_tensor")
            x = tf.reshape(x, [-1, 100, 100, 100, 1])
            x = tf.identity(x, name="input_tensor_after")
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.maxPool1(x)
            x = self.conv4(x)
            x = self.maxPool2(x)
            x = self.conv5(x)
            x = self.maxPool3(x)
            x = self.flatten(x)
            x = self.dense1(x)
            x = self.dropout1(x)
            x = self.dense2(x)
            x = self.dropout2(x)
            return self.dense3(x)


def model_fn(features, labels, mode, params):
    m = CNN_Model()
    logits = CNN_Model.cnn_model(m, features)
    y_pred = tf.nn.softmax(logits=logits)
    y_pred = tf.identity(y_pred, name="output_pred")
    y_pred_cls = tf.argmax(y_pred, axis=1)
    y_pred_cls = tf.identity(y_pred_cls, name="output_cls")

    if mode == tf.estimator.ModeKeys.PREDICT:
        spec = tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=y_pred_cls)
    else:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
                                                                       logits=logits)
        loss = tf.reduce_mean(cross_entropy)

        optimizer = tf.compat.v1.train.AdagradOptimizer(learning_rate=params["learning_rate"])
        train_op = optimizer.minimize(
            loss=loss, global_step=tf.compat.v1.train.get_global_step())
        metrics = {
            "accuracy": tf.metrics.Accuracy()
        }

        spec = tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=metrics)

    return spec


########################################################################################
model = tf.estimator.Estimator(model_fn=model_fn,
                               params={"learning_rate": 0.001},
                               model_dir="./model5/")
########################################################################################
count = 0
while count < 50:
    model.train(input_fn=train_input, steps=92)
    sys.stdout.flush()
    count = count + 1

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
