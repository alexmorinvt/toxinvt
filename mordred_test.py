from scipy.stats.stats import zscore
import tensorflow as tf
import pandas as pd
import numpy as np
from keras import backend as K

from rdkit import Chem
from mordred import Calculator, descriptors

def leaky_relu(z):
    return np.maximum(0.01 * z, z)

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

class LearningRateReducerCb(tf.keras.callbacks.Callback):

  def on_epoch_end(self, epoch, logs={}):
    old_lr = self.model.optimizer.lr.read_value()
    new_lr = old_lr * 0.97
    # print("\nEpoch: {}. Reducing Learning Rate from {} to {}".format(epoch, old_lr, new_lr))
    self.model.optimizer.lr.assign(new_lr)

df = pd.read_table('./TestData/balanced_p53.csv', sep=',')
df.fillna(0, inplace=True)
df = df.sample(frac=1)

#We want: SMILES, PREFERRED_NAME, ACTIVE, and [DESCRIPTORS] as headers
headers = ["SMILES", "PREFERRED_NAME", "ACTIVE"]

calc = Calculator(descriptors, ignore_3D = True)

smiles = []
activity = []

#Creating smiles names and activity lists
for index, molecule in df.iterrows():
    smiles.append(molecule['SMILES'])
    activity.append(molecule['ACTIVE'])

def linearNormalization(np_array):
    returnArr = np.add(np_array, 0.0000000001)
    return returnArr / (returnArr.max(axis=0) + 0.00000001)

def zscoreNormalization(np_array):
    from scipy.stats import zscore
    np_array = zscore(np_array, axis = 0)
    np_array = np.nan_to_num(np_array).astype(int)    
    return np_array

def featureScaling(np_array):
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    np_array = sc.fit_transform(np_array)
    return np_array

def pcaProcessing(train_x, n):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n)
    pca.fit(train_x)
    return pca.transform(train_x)
# Useful code if you want to use mordred to pickle other files
# # As df for training
# activity = pd.DataFrame(activity)
# df.to_pickle(path = "./TestData/activity_p53.pkl", compression='infer', protocol=5, storage_options=None)

# # WARNING: calc.pandas will pull 90%+ CPU Usage for several seconds to several minutes - uses all but one CPU. Be careful not to overheat
# if __name__ == "__main__":
#     #Creating df for descriptors
#     mols = [Chem.MolFromSmiles(smi) for smi in smiles]
#     df = calc.pandas(mols)
#     df.insert(loc=0, column='Active', value=activity)
#     df.insert(loc=0, column='SMILES', value=smiles)

#     df.replace([np.inf, -np.inf], np.nan, inplace=True)
#     df.fillna(0, inplace=True)

#     df.to_pickle(path = "./TestData/pickled_p53.pkl", compression='infer', protocol=5, storage_options=None)

# csv = False
# df = pd.read_pickle("./TestData/pickled_p53.pkl")
df = pd.read_csv("./TestData/balanced_p53.csv")
csv = True

df = df.sample(frac=1)

data = df.values


if csv:
    labels, activity, des = np.split(data, [3, 4], axis = 1)
else:    
    #axis = 1 signifies a vertical split (column wise)
    labels, activity, des = np.split(data, [1, 2], axis = 1)

#Processing np array - changing to float, normalizing, and removing nan
activity = np.asarray(activity).astype('float64')
activity = np.nan_to_num(activity).astype(int)
des = np.asarray(des).astype('float64')
des = np.nan_to_num(des).astype(int)

#Normalization 
des = featureScaling(des)
N = 8
des = pcaProcessing(des, 5)
des = featureScaling(des)

train_labels, test_labels = np.split(labels, [int(.75 * len(labels))], axis = 0)
train_activity, test_activity = np.split(activity, [int(.75 * len(activity))], axis = 0)
train_descriptors, test_descriptors = np.split(des, [int(.75 * len(des))], axis = 0)

model = tf.keras.models.Sequential([
   tf.keras.layers.Dense(128, activation='leaky_relu'),
   tf.keras.layers.Dense(128, activation='leaky_relu'),
   tf.keras.layers.Dense(128, activation='sigmoid'),
   tf.keras.layers.Dense(1, activation = 'sigmoid')
 ])

#standard values are lr = 0.001, b1 = 0.9, b2 = 0.999., ep = 1e-07, amsgrad = False
opt = tf.keras.optimizers.Adam(
    learning_rate=0.000055, beta_1=0.95, beta_2=0.999, epsilon=1e-10, amsgrad=False,
    name='e',
)

class_weight = {0: 1.,
            1: 1.}

model.compile(optimizer=opt,
              loss='binary_crossentropy',
              metrics=['binary_accuracy', f1])

# callbacks=[LearningRateReducerCb()], 
model.fit(train_descriptors, train_activity, callbacks=[LearningRateReducerCb()], 
    class_weight = class_weight, epochs=100, verbose =2)

model.evaluate(test_descriptors, test_activity, verbose=2)

predictions = model.predict(test_descriptors)
predictions = (predictions[:,0] > 0.5).astype(np.int).ravel()
test_activity = test_activity.ravel()

print(tf.get_static_value(tf.math.confusion_matrix(tf.convert_to_tensor(predictions), test_activity)))