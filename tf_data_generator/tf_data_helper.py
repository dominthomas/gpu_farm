import tensorflow as tf
from tensorflow import keras as K
import numpy as np
import nibabel


class DatagenaratorTF():

    def __init__(self, files, labels, batch_size=5, n_channels=1,
                 n_classes=2, shuffle=True, epochs=5):
        """Initialization"""
        self.files = files
        self.labels = labels
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self._EPOCHS = epochs
        self._NUM_CLASSES = n_classes
        self._BATCH_SIZE = batch_size

    def create_tf_dataset(files, labels):
        dataset = tf.data.Dataset.from_tensor_slices(files)
        dataset = dataset.map(load_mri, num_parallel_calls=8)

    def load_mri(file, labels):
        nifti = np.asarray(nibabel.load(file).get_fdata())
        xs, ys, zs = np.where(nifti != 0)
        nifti = nifti[min(xs):max(xs) + 1, min(ys):max(ys) + 1, min(zs):max(zs) + 1]
        nifti = nifti[0:100, 0:100, 0:100]
        nifti = np.reshape(nifti, (100, 100, 100, 1))

        label = K.utils.to_categorical(self.labels[file], num_classes=self.n_classes)
        return nifti, labels
