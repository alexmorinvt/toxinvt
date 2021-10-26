import tensorflow as tf
import pandas as pd
import numpy as np
from scipy import stats

from keras.models import Sequential
from keras.layers import Dense

arr = np.array([[1, 2, 3], [7, 6, 3], [4, 5, -1]])

print(stats.zscore(arr, axis = 0))

