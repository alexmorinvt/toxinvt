from turtle import xcor
import pandas as pd
import numpy as np
import argparse

#MAPS tests to file name and location
file_map = {"test":"../TestData/antibiotic_activity.csv"}

def main():
    return 0

#Returns a map with each test_name : (train_data, test_x, test_y) after preparing data. Keeps the columns NAME, SMILES, SELFIES, TOXICITY, then everything after.
def load_dfs(tests, light, split):
    map = {}

    x_cols = ['NAME', 'SMILES', 'SELFIES']
    tox_col = ['TOXICITY']
    all_x = ['NAME', 'SMILES', 'SELFIES', 'TOXICITY']

    #Make sure there are no extraneous columns in df TODO: Make more flexible
    for test_name in tests:
        df = pd.read_csv(file_map[test_name])
        
        train_length = int(df.shape[0] * split)
        train = df.iloc[:train_length,:]
        test = df.iloc[train_length:,:]

        train_x = train[x_cols]
        train_y = train[tox_col]

        test_x = test[x_cols]
        test_y = test[tox_col]

        if light:
            map[test_name] = {"train":(train_x, train_y), "test":(test_x), "harness":(test_y)}
        else:
            train_descriptors = df.loc[:, ~df.columns.isin(all_x)]
            map[test_name] = {"train":(train_x, train_y, train_descriptors), "test":(test_x), "harness":(test_y)}
        
    return map

if __name__ == "__main__":

    import argparse
    # import sys
    # sys.path.append("./mnist_test.py")

    #Import your file as such, add path if necessary
    from unit_testing import Tester

    #train method must be of the form train([chemical name, SMILES, SELFIES, toxicity], [descriptors])
    #test method must be of the form test([chemical name, SMILES, SELFIES], [descriptors])

    model = Tester()

    descriptors = True 

    parser = argparse.ArgumentParser(description='Testing Specifications')
    parser.add_argument('-a', default = False, action='store_true', help='Runs the entire test suite')
    parser.add_argument('-b', default = False, action='store_true', help='Runs only the balanced tests')
    parser.add_argument('-t', nargs='*', default = ["p53"], help='Specify the tests to be ran')
    parser.add_argument('-v', default = False, action='store_true', help='If specified, print out info on training data')
    parser.add_argument('-save', default = False, action  = 'store_true', help='Saves the each output data to a file named: \'test\' + time.csv')
    parser.add_argument('-split', default = 0.75, help='Percentage to be used as train/test split')

    # -light sends over [name, SMILES, SELFIES, [empty], toxicity]
    parser.add_argument('-light', default = False, action='store_true', help='Does not send descriptors to train to save space')
    parser.add_argument('-test', dest='test_run', action='store_true', help='Only for testing the harness')

    args = parser.parse_args()

    #Verbose setting
    verbose = args.v
    balanced = args.b
    all_tests = args.a
    no_descriptors = args.light
    save = args.save

    tests = args.t

    if args.test_run:
        tests = ['test']

    df_map = load_dfs(tests, args.light, args.split)

    for test_name, test_map in df_map.items():

        train_y = model.train(test_map["train"])
        test_y = model.test(test_map["test"])
        
