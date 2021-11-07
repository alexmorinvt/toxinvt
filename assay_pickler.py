import csv
from os import cpu_count
from rdkit import Chem
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator as RDKitCalculator
#RDKitDescriptors.MolecularDescriptorCalculator(list of descriptors)
from rdkit.Chem import Descriptors as RDKitDescriptors
from mordred import Calculator as MordredCalc
from mordred import descriptors as mordredDescriptors

import getopt, sys

argList = sys.argv[1:]

options = "arwd:"
longOptions = ["Assay", "Read", "Write", "Descriptor"]

#Options.
# -a assayName; default = ACEA_AR_antagonist_80hr
# -r file; default = "./Data/toxcast_large.csv"
# -w file; default = "./Data/'assayName' + _ + 'descriptor source'.pkl"
# -d descriptor; default = RDKit.  Options = RDKit, Mordred

assayName = "ACEA_AR_antagonist_80hr"
read = "'./Data/mordred.pkl"
descriptor = "RDKit"
descriptors = RDKitDescriptors
Calculator = RDKitCalculator
write = "./Data/" + assayName + "_" + descriptor + ".pkl"

try:
    arguments, values = getopt.getopt(argList, options, longOptions)

    for currentArg, currentValue in arguments:
        if currentArg in ("-a", "--Assay"):
            assayName = currentValue
        elif currentArg in ("r", "--Read"):
            read = currentValue
        elif currentArg in ("w", "--Write"):
            write = currentValue
        elif currentArg in ("d", "--Descriptor"):
            if currentValue.casefold == ("Mordred").casefold():
                descriptor = "Mordred:"
                descriptors = mordredDescriptors
                Calculator = MordredCalc
        else:
            print("Invalid argument: ", currentArg, currentValue)

except getopt.error as err:
    print(str(err))

if descriptor == "Mordred":
    calc = Calculator(descriptors, ignore_3D = True)
