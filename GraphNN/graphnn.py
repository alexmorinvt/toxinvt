import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from rdkit import Chem
from rdkit import RDLogger
# from rdkit.Chem.Draw import IPythonConsole
# from rdkit.Chem.Draw import MolsToGridImage
import logging
from chemutils import graphs_from_smiles, MPNNModel, MPNNDataset

tf.get_logger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

# np.random.seed(42)
# tf.random.set_seed(42)

df = pd.read_table('../TestData/balanced_p53.csv', sep=',')
df.fillna(0, inplace=True)
df = df.sample(frac=1)
df = df[["SMILES", "ACTIVE"]]

# Shuffle array of indices ranging from 0 to 2049
permuted_indices = np.random.permutation(np.arange(df.shape[0]))

# Train set: 80 % of data
train_index = permuted_indices[: int(df.shape[0] * 0.8)]
x_train = graphs_from_smiles(df.iloc[train_index].SMILES)
y_train = df.iloc[train_index].ACTIVE

# Valid set: 10 % of data
valid_index = permuted_indices[int(df.shape[0] * 0.8) : int(df.shape[0] * 0.9)]
x_valid = graphs_from_smiles(df.iloc[valid_index].SMILES)
y_valid = df.iloc[valid_index].ACTIVE

# Test set: 10 % of data
test_index = permuted_indices[int(df.shape[0] * 0.9) :]
x_test = graphs_from_smiles(df.iloc[test_index].SMILES)
y_test = df.iloc[test_index].ACTIVE

mpnn = MPNNModel(
    atom_dim=x_train[0][0][0].shape[0], bond_dim=x_train[1][0][0].shape[0],
)

mpnn.compile(
    loss=keras.losses.BinaryCrossentropy(),
    optimizer=keras.optimizers.Adam(),
    metrics=[keras.metrics.AUC(name="AUC"), 'binary_accuracy'],
)

# keras.utils.plot_model(mpnn, show_dtype=True, show_shapes=True)

train_dataset = MPNNDataset(x_train, y_train)
valid_dataset = MPNNDataset(x_valid, y_valid)
test_dataset = MPNNDataset(x_test, y_test)

history = mpnn.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=50,
    verbose=2,
    # class_weight={0: 2.0, 1: 0.5},
)

plt.figure(figsize=(10, 6))
plt.plot(history.history["AUC"], label="train AUC")
plt.plot(history.history["val_AUC"], label="valid AUC")
plt.xlabel("Epochs", fontsize=16)
plt.ylabel("AUC", fontsize=16)
plt.legend(fontsize=16)

mpnn.evaluate(test_dataset, verbose=2)

# predictions = mpnn.predict(x_test)
# predictions = (predictions[:,0] > 0.5).astype(np.int).ravel()
# y_test = y_test.ravel()

# print(tf.get_static_value(tf.math.confusion_matrix(tf.convert_to_tensor(predictions), y_test)))