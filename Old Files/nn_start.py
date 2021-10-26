import pandas as pd
import numpy as np

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

df = pd.read_table('./Data/TOX21_p53_BLA_p1_ch1.csv', sep=',')
df.fillna(0, inplace=True)

data = df.values

#axis = 1 signifies a vertical split (column wise)
labels, activity, descriptors = np.split(data, [3, 4], axis = 1)

x_train, x_test, y_train, y_test = train_test_split(descriptors, activity, test_size=0.2, random_state=42)
x_train = np.asarray(x_train).astype('float32')
x_test = np.asarray(x_test).astype('float32')
y_train = np.asarray(y_train).astype('float32')
y_test = np.asarray(y_test).astype('float32')


#There are 208 descriptors
model = Sequential()
model.add(Dense(20, activation = "relu", input_shape = (208,)))
model.add(Dense(20, activation="relu"))
model.add(Dense(1, activation='sigmoid'))

sgd = tf.keras.optimizers.SGD(learning_rate=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(x_train, y_train,epochs=1, batch_size=1, verbose=1)

y_pred = model.predict(x_test)
print(y_pred)

score = model.evaluate(x_test, y_test, verbose=1)

active_score = [0, 0]
inactive_score = [0, 0]


for i in len(y_pred):
    if y_pred[i] == 0:
        inactive_score[1] += 1
        if y_pred[i] == y_test[i]:
            inactive_score[0] += 1
    
    else:
        active_score[1] += 1
        if y_pred[i] == y_test[i]:
            inactive_score[0] += 1

print("Inactive Accuracy:", inactive_score[0] / inactive_score[1])
print("Active Accuracy:", active_score[0] / active_score[1])