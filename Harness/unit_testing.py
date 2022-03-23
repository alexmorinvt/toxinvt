import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import keras
import csv

from tensorflow import nn
import tensorflow as tf

import pandas as pd

# df = pd.read_table('../TestData/antibiotic_activity.csv', sep=',')

ayo = -1

class LearningRateReducerCb(tf.keras.callbacks.Callback):

  def on_epoch_end(self, epoch, logs={}):
    old_lr = self.model.optimizer.lr.read_value()
    new_lr = old_lr * 0.999
    print("\nEpoch: {}. Reducing Learning Rate from {} to {}".format(epoch, old_lr, new_lr))
    self.model.optimizer.lr.assign(new_lr)

class Tester:
    def __init__(self) -> None:
        self.ayo = -1
        self.model = None
        pass

    def train(self, test_name, tup):

        names, y_train, x_train = tup

        self.model = keras.Sequential([
            keras.layers.Flatten(),
            keras.layers.Dense(16, activation=nn.relu),
            keras.layers.Dense(64, activation=nn.relu),
            keras.layers.Dense(64, activation=nn.relu),
            keras.layers.Dense(16, activation=nn.relu),
            keras.layers.Dense(1, activation=nn.sigmoid),
        ])

        x_train = x_train.replace(np.nan, -1)

        print(x_train.shape)
        print(y_train.shape)

        #standard values are lr = 0.001, b1 = 0.9, b2 = 0.999., ep = 1e-07, amsgrad = False
        opt = tf.keras.optimizers.Adam(
            learning_rate=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
            name='Adam',
        )

        self.model.compile(optimizer=opt,
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

        class_weight = {0: 1.,
                        1: 0.66}

        self.model.fit(x_train, y_train, callbacks=[LearningRateReducerCb()], epochs=100, class_weight = class_weight)
        return 0

    def test(self, test_name, tup):
        names, x_test = tup

        predictions = self.model.predict(x_test)
        predictions = (predictions[:,0] > 0.5).astype(np.int).ravel()
        return predictions
