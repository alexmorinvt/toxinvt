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

df = pd.read_table('./TestData/aid2538.csv', sep=',')

with open('./TestData/aid2538.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)

        fieldnames = ['COUNT', 'SMILES', "ACTIVE"]

        #Gets names of the Descriptors for 1. headers and 2. methods
        descriptor_names = [x[0] for x in Descriptors._descList]
        descriptor_methods = []

        #Creating a list of methods
        for name in descriptor_names:
            method = getattr(Descriptors, name, 0)
            descriptor_methods.append(method)
            fieldnames.append(name)

        #Line headers: ,ASSAY_NAME,CASRN,SMILES,DTXSID,PREFERRED_NAME,HIT_CALL,FLAGS
        count = 1

        with open('./TestData/aid2538_des', "w", newline = '') as csvfile:
            writer = csv.writer(csvfile, delimiter =',')
            writer.writerow(fieldnames)          

            #Reads through all molecules - if match, run descriptors and save row  
            for row in reader:

                descriptor_results = [count, row['CID'], row['ACTIVITY']]

                mol = Chem.MolFromCID(row['CID'])
                for method in descriptor_methods:
                    descriptor_results.append(method(mol))
                
                writer.writerow(descriptor_results)

