# make sure  numpy, scipy, pandas, sklearn are installed, otherwise run
# pip install numpy scipy pandas scikit-learn
import numpy as np
import pandas as pd
from scipy import io
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from rdkit import Chem
import gzip
import selfies as sf

suppl = Chem.rdmolfiles.ForwardSDMolSupplier(gzip.open('tox21.sdf.gz'))

# load data

y_tr = pd.read_csv('tox21_labels_train.csv.gz', index_col=0, compression="gzip")
y_te = pd.read_csv('tox21_labels_test.csv.gz', index_col=0, compression="gzip")
x_tr_dense = pd.read_csv('tox21_dense_train.csv.gz', index_col=0, compression="gzip").values
x_te_dense = pd.read_csv('tox21_dense_test.csv.gz', index_col=0, compression="gzip").values
x_tr_sparse = io.mmread('tox21_sparse_train.mtx.gz').tocsc()
x_te_sparse = io.mmread('tox21_sparse_test.mtx.gz').tocsc()

train_num_molecules = x_tr_sparse.shape[0]
test_num_molecules = y_tr.shape[0]
import sys

x = 1
smiles = []
selfies = []
names = []
for mol in suppl:
    if mol is None:
        continue

    if x > train_num_molecules:
        break
    smile = Chem.rdmolfiles.MolToSmiles(mol)
    smiles.append(smile)
    names.append(smile)
    try:
        selfie = sf.encoder(smile)
    except sf.EncoderError:
        selfie = np.nan
    selfies.append(selfie)
    x += 1

print("test len:", x - 1)
print(len(smiles), len(selfies))

smiles_selfies_train = pd.DataFrame(zip(names, smiles, selfies), columns = pd.Series(['NAMES', 'SMILES', 'SELFIES']))

x = 1
smiles = []
selfies = []
names = []
for mol in suppl:
    if mol is None:
        continue

    if x > train_num_molecules:
        break
    smile = Chem.rdmolfiles.MolToSmiles(mol)
    smiles.append(smile)
    names.append(smile)
    try:
        selfie = sf.encoder(smile)
    except sf.EncoderError:
        selfie = np.nan
    selfies.append(selfie)
    x += 1
print("test len:", x - 1)

smiles_selfies_test = pd.DataFrame(zip(names, smiles, selfies), columns = pd.Series(['NAMES', 'SMILES', 'SELFIES']))
smiles_selfies_test = smiles_selfies_test[['NAMES', 'SMILES', 'SELFIES']]

# filter out very sparse features
sparse_col_idx = ((x_tr_sparse > 0).mean(0) > 0.05).A.ravel()
x_tr = np.hstack([x_tr_dense, x_tr_sparse[:, sparse_col_idx].A])
x_te = np.hstack([x_te_dense, x_te_sparse[:, sparse_col_idx].A])

test_map = {"train":(smiles_selfies_train, y_tr, x_tr), "test":(smiles_selfies_test, x_te), "harness":(y_te)}

print(test_map['train'][0].head)
print(test_map['test'][0].head)

sys.exit()


count = 0
for mol in suppl:
    count += 1
    if mol is not None:
        print(Chem.rdmolfiles.MolToSmiles(mol))


print(count)

def converter(file_name):
    sppl = Chem.SDMolSupplier(file_name)
    outname = file_name.replace(".sdf", ".txt")
    out_file = open(outname, "w")
    for mol in sppl:
        if mol is not None:# some compounds cannot be loaded.
            smi = Chem.MolToSmiles(mol)
            name = mol.GetProp("_Name")
            out_file.write(f"{smi}\t{name}\n")
    out_file.close()
if __name__ == "__main__":
    converter(sys.argv[1])