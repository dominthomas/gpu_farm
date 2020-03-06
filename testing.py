import numpy as np
import random

random_list = ["asdf1", "asdf2", "asdf3", "asdf4", "asdf5"]
random_labels = [1, 2, 1, 2, 1]

random_list2 = ["bob1", "bob2", "bob3", "bob4", "bob5"]
random_labels2 = {'first': 0, 'second': 2, 'third': 0, 'fourth': 2, 'fifth': 0}

random.Random(129).shuffle(random_list)
random.Random(129).shuffle(random_labels)
random.Random(129).shuffle(random_list2)
random.Random(129).shuffle(random_labels2)

print(random_list)
print(random_labels)
print(random_list2)
print(random_labels2)

random_list_dictionary = {'train': random_list, 'validation': random_list2}
print(random_list_dictionary)

random_list_labels = {}
for item in random_list:
    random_list_labels[item] = 1

random.Random(129).shuffle(random_labels)
random.Random(129).shuffle(random_labels2)


print(random_labels)
print(random_labels2)








########################################################################################
class CnnModel(Model):
    with tf.device("/cpu:0"):
        def __init__(self,
                     loss_object,
                     optimizer,
                     train_loss,
                     train_metric,
                     test_loss,
                     test_metric):
            super(CnnModel, self).__init__()
            with tf.device("/gpu:0"):
                self.conv1 = Conv3D(64,
                                    input_shape=(100, 100, 100, 1),
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

                self.loss_object = loss_object
                self.optimizer = optimizer
                self.train_loss = train_loss
                self.train_metric = train_metric
                self.test_loss = test_loss
                self.test_metric = test_metric


########################################################################################