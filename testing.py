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
########################################################################################

class cn_model(Model):
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
                    self.dense3 = Dense(1, activation='sigmoid')

                self.loss_object = loss_object
                self.optimizer = optimizer
                self.train_loss = train_loss
                self.train_metric = train_metric
                self.test_loss = test_loss
                self.test_metric = test_metric

        def cnn_model(self, x):
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
            x = self.dense1(x)
            x = self.dropout1(x)
            return self.dense3(x)


########################################################################################

########################################################################################

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


########################################################################################
def cnn_model(features, labels, mode, params):
    with tf.device("/cpu:0"):
        with tf.device("/gpu:0"):
            net = features
            net = tf.identity(net, name="input_tensor")

            net = Conv3D(64,
                         input_shape=(100, 100, 100, 1),
                         data_format='channels_last',
                         kernel_size=(7, 7, 7),
                         strides=(2, 2, 2),
                         padding='valid',
                         activation='relu')

        with tf.device("/gpu:1"):
            net = Conv3D(64,
                         kernel_size=(3, 3, 3),
                         padding='valid',
                         activation='relu')

        with tf.device("/gpu:2"):
            net = Conv3D(128,
                         kernel_size=(3, 3, 3),
                         padding='valid',
                         activation='relu')

            net = MaxPooling3D(pool_size=(2, 2, 2),
                               padding='valid')

        with tf.device("/gpu:3"):
            net = Conv3D(128,
                         kernel_size=(3, 3, 3),
                         padding='valid',
                         activation='relu')

            net = MaxPooling3D(pool_size=(2, 2, 2),
                               padding='valid')

        with tf.device("/gpu:4"):
            net = Conv3D(128,
                         kernel_size=(3, 3, 3),
                         padding='valid',
                         activation='relu')

            net = MaxPooling3D(pool_size=(2, 2, 2),
                               padding='valid')

            net = Flatten()

            net = Dense(units=256, activation='relu')
            net = Dropout(0.7)
            net = Dense(units=256, activation='relu')
            net = Dropout(0.7)
            net = Dense(units=1, activation='sigmoid')

            logits = net
            y_pred = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
            y_pred = tf.identity(y_pred, name="output_pred")
            y_pred_cls = tf.argmax(y_pred, axis=1)
            y_pred_cls = tf.identity(y_pred_cls, name="output_cls")

    if mode == tf.estimator.ModeKeys.PREDICT:
        spec = tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=y_pred_cls)
    else:
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,
                                                                logits=logits)
        loss = tf.reduce_mean(cross_entropy)

        optimizer = tf.compat.v1.train.AdagradOptimizer(learning_rate=params["learning_rate"])
        train_op = optimizer.minimize(
            loss=loss, global_step=tf.train.get_global_step())
        metrics = {
            "accuracy": tf.metrics.accuracy(labels, y_pred_cls)
        }

        spec = tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=metrics)

    return spec


########################################################################################
