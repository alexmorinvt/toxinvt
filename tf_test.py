from tensorflow import keras
from tensorflow import nn
import tensorflow as tf
import pandas as pd
import numpy as np

import pandas as pd
df = pd.read_csv('./TestData/antibiotic_activity.csv')
df = df.sample(frac=1)

class LearningRateReducerCb(tf.keras.callbacks.Callback):

  def on_epoch_end(self, epoch, logs={}):
    old_lr = self.model.optimizer.lr.read_value()
    new_lr = old_lr * 0.999
    print("\nEpoch: {}. Reducing Learning Rate from {} to {}".format(epoch, old_lr, new_lr))
    self.model.optimizer.lr.assign(new_lr)

properties = list(df.columns.values)
properties.remove('Activity')
X = df[properties].astype('float64')
y = df['Activity'].astype('float64')

x_train, x_test = np.split(X, [int(0.75 * len(X))], axis = 0)
y_train, y_test = np.split(y, [int(0.75 * len(y))], axis = 0)

print(x_train.head())

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(4,)),
    keras.layers.Dense(16, activation=nn.relu),
	keras.layers.Dense(64, activation=nn.relu),
    keras.layers.Dense(64, activation=nn.relu),
    keras.layers.Dense(16, activation=nn.relu),
    keras.layers.Dense(1, activation=nn.sigmoid),
])

#standard values are lr = 0.001, b1 = 0.9, b2 = 0.999., ep = 1e-07, amsgrad = False
opt = tf.keras.optimizers.Adam(
    learning_rate=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
    name='Adam',
)

model.compile(optimizer=opt,
              loss='binary_crossentropy',
              metrics=['accuracy'])

class_weight = {0: 1.,
                1: 0.66}

model.fit(x_train, y_train, callbacks=[LearningRateReducerCb()], epochs=5000, class_weight = class_weight, verbose = 0)
test_loss, test_acc = model.evaluate(x_test, y_test)

model.evaluate(x_test, y_test, verbose=2)
predictions = model.predict(x_test)
predictions = (predictions[:,0] > 0.5).astype(np.int).ravel()

test_activity = y_test.ravel()

# x = 0
# for pred in predictions:
#     print(pred, test_activity[x])
#     x+=1

print(tf.math.confusion_matrix(tf.convert_to_tensor(predictions), test_activity))