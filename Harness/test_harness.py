import pandas as pd
import numpy as np
import argparse
from sklearn import metrics
import argparse
import sys
from os import devnull

#MAPS tests to file name and location
file_map = {
    "TOX21_AhR_LUC_Agonist_viability  ":"../HarnessData/TOX21_AhR_LUC_Agonist_viability.csv",
    "TOX21_AhR_LUC_Agonist ":"../HarnessData/TOX21_AhR_LUC_Agonist.csv",
    "TOX21_AR_BLA_Agonist_ratio":"../HarnessData/TOX21_AR_BLA_Agonist_ratio.csv",
    "TOX21_AR_LUC_MDAKB2_Antagonist_10nM_R1881_viability":"../HarnessData/TOX21_AR_LUC_MDAKB2_Antagonist_10nM_R1881_viability.csv",
    "TOX21_ARE_BLA_agonist_ratio":"../HarnessData/TOX21_ARE_BLA_agonist_ratio.csv",
    "TOX21_Aromatase_Inhibition":"../HarnessData/TOX21_Aromatase_Inhibition.csv",
    "TOX21_CAR_Agonist":"../HarnessData/TOX21_CAR_Agonist.csv",
    "TOX21_ERa_BLA_Agonist_ratio":"../HarnessData/TOX21_ERa_BLA_Agonist_ratio.csv",
    "TOX21_ERb_BLA_Agonist_ratio":"../HarnessData/TOX21_ERb_BLA_Agonist_ratio.csv",
    "TOX21_ERR_Agonist":"../HarnessData/TOX21_ERR_Agonist.csv",
    "TOX21_NFkB_BLA_agonist_ratio":"../HarnessData/TOX21_NFkB_BLA_agonist_ratio.csv",
    "TOX21_p53_BLA_p1_ratio":"../HarnessData/TOX21_p53_BLA_p1_ratio.csv",
    "TOX21_p53_BLA_p1_viability":"../HarnessData/TOX21_p53_BLA_p1_viability.csv"}
rng = None

# Disable
def blockPrint():
    sys.stdout = open(devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

#Returns a map with each test_name : (train_data, test_data, test_y) after preparing data. Keeps the columns NAME, SMILES, SELFIES, TOXICITY, then everything after.
def load_dfs(tests, light, split, balanced):
    map = {}

    x_cols = ['NAME', 'SMILES', 'SELFIES']
    tox_col = ['TOXICITY']
    all_x = ['NAME', 'SMILES', 'SELFIES', 'TOXICITY']

    #Make sure there are no extraneous columns in df TODO: Make more flexible
    for test_name in tests:
        df = pd.read_csv(file_map[test_name]).sample(frac=1)
        
        #Produces a balanced df based on TOXICITY
        if balanced:
            df = df.groupby('TOXICITY')
            df = df.apply(lambda x: x.sample(df.size().min()).reset_index(drop=True))
            df = df.sample(frac=1)

        #Splitting data
        train_length = int(df.shape[0] * split)
        train = df.iloc[:train_length,:]
        test = df.iloc[train_length:,:]

        train_x = train[x_cols]
        train_y = train[tox_col]

        test_x = test[x_cols]
        test_y = test[tox_col]

        #Pass on descriptors?
        if light:
            map[test_name] = {"train":(train_x, train_y), "test":(test_x), "harness":(test_y)}
        else:
            train_descriptors = train.loc[:, ~train.columns.isin(all_x)]
            test_descriptors = test.loc[:, ~test.columns.isin(all_x)]
            map[test_name] = {"train":(train_x, train_y, train_descriptors), "test":(test_x, test_descriptors), "harness":(test_y["TOXICITY"].values)}
        
    return map

def main():

    # Import your file as such, add path if necessary

    # sys.path.append("./mnist_test.py")

    # Class needs two methods, train and test
    # Train is of the form train( (x, y, descriptors) ) or ( (x, y) )
    # Test is of the form test( (x, descriptors) )
    # For both, descriptors is optional in the passed tuple. Look for the -light argument
    from unit_testing import Tester
    model = Tester()

    from time import strftime
    
    #train method must be of the form train([chemical name, SMILES, SELFIES, toxicity], [descriptors])
    #test method must be of the form test([chemical name, SMILES, SELFIES], [descriptors])

    descriptors = True 

    parser = argparse.ArgumentParser(description='Testing Specifications')
    parser.add_argument('-a', default = False, action='store_true', help='Runs the entire test suite')
    parser.add_argument('-b', default = False, action='store_true', help='Runs only the balanced tests')
    parser.add_argument('-t', nargs='*', default = ["TOX21_p53_BLA_p1_ratio"], help='Specify the tests to be ran')
    parser.add_argument('-v', default = False, action='store_true', help='If specified, print out info on training data')
    parser.add_argument('-save', default = False, action  = 'store_true', help='Saves the each output data to a file named: \'test\' + time.csv')
    parser.add_argument('-n', default = 1, help='Number of repititions for each model')
    parser.add_argument('-split', default = 0.75, help='Percentage to be used as train/test split')
    parser.add_argument('-seed', default=42, help='Seed for randomization')
    parser.add_argument('-light', default = False, action='store_true', help='Does not send descriptors to train to save space')
    parser.add_argument('-test', dest='test_run', action='store_true', help='Only for testing the harness')
    parser.add_argument('-m', nargs='*', default = ["report", "confusion_matrix"], help='Specify the metrics to be ran')

    args = parser.parse_args()

    np.random.seed(args.seed) #Seed Done
    #Use the random number generator if you'd like
    rng = np.random.default_rng(args.seed)

    #Verbose setting
    verbose = args.v #Done
    balanced = args.b  #Done
    all_tests = args.a #Done
    no_descriptors = args.light #Done
    save = args.save #Done
    split = args.split #Done
    metrs = args.m #Done

    tests = args.t #Done, TODO: Add tests

    if all_tests:
        test = file_map.keys()
        
    if args.test_run: #Done
        file_map["test"] = "../HarnessData/antibiotic_activity.csv"
        tests = ['test']

    df_map = load_dfs(tests, no_descriptors, split, balanced)

    print_output = ''
    #Actual loop that loops through the test map
    for test_name, test_map in df_map.items():

        if not verbose:
            blockPrint()    

        train_y = model.train(test_name, test_map["train"])
        test_predictions = model.test(test_name, test_map["test"])

        test_y = test_map["harness"]

        if not verbose:
            enablePrint()

        #Metrics control here
        #test_predictions should be a vector of the shape (test_length,)

        #I love this metrics package - so simple
        current_metrics = metrics.classification_report(test_y, test_predictions, target_names = ['non-toxic', 'toxic'], digits = 4)

        accuracy = metrics.accuracy_score(test_y, test_predictions)
        f1score = metrics.f1_score(test_y, test_predictions)

        precision = metrics.precision_score(test_y, test_predictions)
        mse = metrics.mean_squared_error(test_y, test_predictions)
        log_l = metrics.log_loss(test_y, test_predictions)
        confusion_matrix = metrics.confusion_matrix(test_y, test_predictions)
        
        metrics_map = {'report':current_metrics, 'accuracy':accuracy, 'precision':precision, 'f1':f1score, 'mse':mse,
            'log_loss':log_l, 'confusion_matrix':confusion_matrix}

        if save:
            test_y = pd.DataFrame(test_y, columns = ['EXPECTED'])
            test_predictions = pd.DataFrame(test_predictions, columns = ['PREDICTED'])

            df = pd.concat([test_map['test'][0].reset_index(drop=True), test_y.reset_index(drop=True), test_predictions.reset_index(drop=True)], 
                axis = 1, ignore_index=True)
            
            df.columns = list(test_map['test'][0].columns) + ['EXPECTED', 'PREDICTED']

            timestr = strftime("%Y%m%d-%H%M%S")

            df.to_pickle(f"../HarnessResults/{test_name}_{timestr}.pkl")
        
        for m in metrs:
            print_output += str(metrics_map[m])

        print("test name: " + test_name)
        for m in metrs:
            print(metrics_map[m])

    timestr = strftime("%Y%m%d-%H%M%S")
    text_file = open(f"../HarnessResults/results_{timestr}.txt", "w")
    text_file.write(print_output)
    text_file.close()
            
        

if __name__ == "__main__":
    main()
