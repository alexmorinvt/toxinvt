import tensorflow as tf
import pandas as pd
import numpy as np
import scipy.stats as stats

#60.1% active: ACEA_AR_antagonist_80hr (1770 sample size)
#51.6%: ATG_PXRE_CIS_up (3522)
#57.8%: BSK_hDFCGF_Proliferation_down (1430)
#52.8%: LTEA_HepaRG_CYP1A2_up (1016)
#47.6%: LTEA_HepaRG_CYP2B6_up (1016)
#52.4%: LTEA_HepaRG_CYP2E1_dn (1016)
#53.3%: LTEA_HepaRG_SLC10A1_dn (1016)
#61.9%: TOX21_DT40 (7948)
#59.6%: TOX21_DT40_100 (7948)
#62.9%: TOX21_DT40_657 (7948)

#All you need to do is change this list to generate new pkl files, provided you have the correct files downloaded
assays = ["ACEA_AR_antagonist_80hr"]#, "ATG_PXRE_CIS_up", "BSK_hDFCGF_Proliferation_down", "LTEA_HepaRG_CYP1A2_up", "LTEA_HepaRG_CYP2B6_up", "LTEA_HepaRG_CYP2E1_dn", "LTEA_HepaRG_SLC10A1_dn", "TOX21_DT40", "TOX21_DT40_100", "TOX21_DT40_657"]

class LearningRateReducerCb(tf.keras.callbacks.Callback):

  def on_epoch_end(self, epoch, logs={}):
    old_lr = self.model.optimizer.lr.read_value()
    new_lr = old_lr * 0.995
    print("\nEpoch: {}. Reducing Learning Rate from {} to {}".format(epoch, old_lr, new_lr))
    self.model.optimizer.lr.assign(new_lr)

def leaky_relu(z):
    return np.maximum(0.01 * z, z)

#Data is in the form [SMILES, ACTIVITY, [descriptors]]. randomized by sample
assay_data = pd.read_csv("./Data/ATG_PXRE_CIS_up.csv")
assay_data = assay_data.sample(frac=1)
properties = list(assay_data.columns.values)

activity = assay_data["ACTIVE"]
properties = list(assay_data.columns.values)
properties.remove('ACTIVE')
properties.remove('COUNT')
properties.remove('SMILES')
properties.remove('PREFERRED_NAME')

descriptors = assay_data[properties].values
descriptors = np.add(descriptors, 0.000000000213)
descriptors = stats.zscore(descriptors, axis=1)


#axis = 1 signifies a vertical split (column wise). Currently set at 90%
train_x, test_x = np.split(descriptors, [int(0.9 * len(descriptors))], axis = 0)
train_y, test_y = np.split(activity, [int(0.9 * len(activity))], axis = 0)

#Processing np array - changing to float, normalizing, and removing nan
train_x = np.asarray(train_x).astype('float32')
test_x = np.asarray(test_x).astype('float32')
train_y = np.asarray(train_y).astype('float32')
test_y = np.asarray(test_y).astype('float32')

print(train_x.shape)
print(train_y.shape)

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(208,)),
    tf.keras.layers.Dense(208, activation='relu'),
    tf.keras.layers.Dense(208, activation='relu'),
    tf.keras.layers.Dense(208, activation='relu'),
    tf.keras.layers.Dense(1, activation = 'sigmoid')
])

#standard values are lr = 0.001, b1 = 0.9, b2 = 0.999., ep = 1e-07, amsgrad = False
opt = tf.keras.optimizers.Adam(
    learning_rate=0.1, beta_1=0.9, beta_2=0.999, epsilon=0.0001, amsgrad=False,
    name='e',
)

# opt = tf.keras.optimizers.SGD(
#     learning_rate=0.01, momentum=0.0, nesterov=False, name="SGD"
# )

class_weight = {0: 1000.,
            1: 1.}

model.compile(optimizer="Adam",
            loss='binary_crossentropy',
            metrics=['accuracy'])

model.fit(train_x, train_y, callbacks=[LearningRateReducerCb()], epochs=5, class_weight = class_weight)

model.evaluate(test_x, test_y, verbose=2)
predictions = model.predict(test_x)
predictions = (predictions[:,0] > 0.5).astype(np.int).ravel()
test_activity = test_y.ravel()

print(tf.math.confusion_matrix(tf.convert_to_tensor(predictions), test_activity))