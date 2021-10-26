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
print(df.shape)

data = df.values
df = df.sample(frac=1)

#axis = 1 signifies a vertical split (column wise)
train, test = np.split(data, [int(9.0 / 10 * len(data))], axis = 0)

train_labels, train_activity, train_descriptors = np.split(train, [1, 2], axis = 1)
test_labels, test_activity, test_descriptors = np.split(test, [1, 2], axis = 1)

for x in range(1, 5):
    print(train_labels[x])
    print(train_activity[x])
    print(test_labels[x])

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

train_descriptors = train_descriptors / (train_descriptors.max(axis=0) + 0.00000001)
test_descriptors = test_descriptors / (test_descriptors.max(axis=0) + 0.00000001)

model = tf.keras.models.Sequential([
   tf.keras.layers.Dense(208, activation='relu'),
   tf.keras.layers.Dense(128, activation='relu'),
   tf.keras.layers.Dense(128, activation='relu'),
   tf.keras.layers.Dense(128, activation='relu'),
   tf.keras.layers.Dropout(0.5),
   tf.keras.layers.Dense(1, activation = 'sigmoid')
 ])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(train_descriptors, train_activity, epochs=100)

model.evaluate(test_descriptors, test_activity, verbose=2)