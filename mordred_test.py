from scipy.stats.stats import zscore
import tensorflow as tf
import pandas as pd
import numpy as np

from rdkit import Chem
from mordred import Calculator, descriptors

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

df = pd.read_pickle("./TestData/pickled_p53.pkl")

df = df.sample(frac=1)

data = df.values

#axis = 1 signifies a vertical split (column wise)
labels, activity, des = np.split(data, [1, 2], axis = 1)

#Processing np array - changing to float, normalizing, and removing nan
activity = np.asarray(activity).astype('float64')
activity = np.nan_to_num(activity).astype(int)
des = np.asarray(des).astype('float64')
des = np.nan_to_num(des).astype(int)

#Normalization 
des = featureScaling(des)
N = 3
des = pcaProcessing(des, N)

train_labels, test_labels = np.split(labels, [int(.75 * len(labels))], axis = 0)
train_activity, test_activity = np.split(activity, [int(.75 * len(activity))], axis = 0)
train_descriptors, test_descriptors = np.split(des, [int(.75 * len(des))], axis = 0)

model = tf.keras.models.Sequential([
   tf.keras.layers.Dense(N * 5, activation='relu'),
   tf.keras.layers.Dense(128, activation='relu'),
   tf.keras.layers.Dense(128, activation='relu'),
   tf.keras.layers.Dense(128, activation='relu'),
   tf.keras.layers.Dropout(0.5),
   tf.keras.layers.Dense(1, activation = 'sigmoid')
 ])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(train_descriptors, train_activity, epochs=50, verbose =2, batch_size = 10)

model.evaluate(test_descriptors, test_activity, verbose=2)

predictions = model.predict(train_descriptors)
predictions = (predictions[:,0] > 0.5).astype(np.int).ravel()
test_activity = train_activity.ravel()

print(tf.math.confusion_matrix(tf.convert_to_tensor(predictions), test_activity))