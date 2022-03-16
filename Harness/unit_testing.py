import tensorflow as tf
import pandas as pd
import numpy as np
from scipy import stats
import csv

from keras.models import Sequential
from keras.layers import Dense

from rdkit import Chem
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Descriptors

# df = pd.read_table('../TestData/antibiotic_activity.csv', sep=',')

ayo = -1

class Tester:
    def __init__(self) -> None:
        self.ayo = -1
        pass

    def train(self, tup):
        print(self.ayo)
        self.ayo = 10
        names, y, descriptors = tup
        return 0

    def test(self, tup):
        print(self.ayo)
        return -1
