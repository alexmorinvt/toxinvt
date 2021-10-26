import tensorflow as tf
import pandas as pd
import numpy as np
from scipy import stats

from keras.models import Sequential
from keras.layers import Dense

df = pd.read_table('./TestData/balanced_p53.csv', sep=',')
df.fillna(0, inplace=True)
df = df.sample(frac=1)

data = df.values

#axis = 1 signifies a vertical split (column wise)
train, test = np.split(data, [int(3.0 / 4 * len(data))], axis = 0)

train_labels, train_activity, train_descriptors = np.split(train, [3, 4], axis = 1)
test_labels, test_activity, test_descriptors = np.split(test, [3, 4], axis = 1)

#Processing np array - changing to float, normalizing, and removing nan
train_activity = np.asarray(train_activity).astype('float64')
train_descriptors = np.asarray(train_descriptors).astype('float64')
test_activity = np.asarray(test_activity).astype('float64')
test_descriptors = np.asarray(test_descriptors).astype('float64')

train_activity = np.nan_to_num(train_activity).astype(int)
train_descriptors = np.nan_to_num(train_descriptors)
test_activity = np.nan_to_num(test_activity).astype(int)
test_descriptors = np.nan_to_num(test_descriptors)

train_descriptors = np.add(train_descriptors, 0.0000000001)
test_descriptors = np.add(test_descriptors, 0.0000000001)

train_activity = train_activity / (train_activity.max(axis=0) + 0.00000001)
train_descriptors = train_descriptors / (train_descriptors.max(axis=0) + 0.00000001)
test_activity = test_activity / (test_activity.max(axis=0) + 0.00000001)
test_descriptors = test_descriptors / (test_descriptors.max(axis=0) + 0.00000001)

model = tf.keras.models.Sequential([
   tf.keras.layers.Dense(208, activation='relu'),
   tf.keras.layers.Dense(128, activation='relu'),
   tf.keras.layers.Dense(128, activation='relu'),
   tf.keras.layers.Dense(128, activation='relu'),
   tf.keras.layers.Dropout(0.4),
   tf.keras.layers.Dense(1, activation = 'sigmoid')
 ])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(train_descriptors, train_activity, epochs=1000)

model.evaluate(test_descriptors, test_activity, verbose=2)