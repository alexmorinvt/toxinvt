import csv
from rdkit import Chem
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Descriptors

#May need to disable this
from multiprocessing import Pool
import os
import threading
import selfies as sf
from numpy import nan
sf.set_semantic_constraints("hypervalent")

#READ ME: this file is not very useful with Professor Reisfeld's pickled files uploaded. However, this can be easily adapted to create assay specific pickled files
def createDBByName(wrapper):
    print("start")

    mydata = threading.local()
    mydata.assay, mydata.inputfile, mydata.outputfile = wrapper

    with open(mydata.inputfile, newline='') as csvfile:
        mydata.reader = csv.DictReader(csvfile)

        mydata.fieldnames = ['NAME', 'SMILES', "SELFIES", "TOXICITY"]

        #Gets names of the Descriptors for 1. headers and 2. methods
        mydata.descriptor_names = [x[0] for x in Descriptors._descList]
        mydata.descriptor_methods = []

        #Creating a list of methods
        for name in mydata.descriptor_names:
            method = getattr(Descriptors, name, 0)
            mydata.descriptor_methods.append(method)
            mydata.fieldnames.append(name)

        #Line headers: ,ASSAY_NAME,CASRN,SMILES,DTXSID,PREFERRED_NAME,HIT_CALL,FLAGS
        count = 1
        with open(mydata.outputfile, "w", newline = '') as csvfile:
            mydata.writer = csv.writer(csvfile, delimiter =',')
            mydata.writer.writerow(mydata.fieldnames)          

            #Reads through all molecules - if match, run descriptors and save row  
            for row in mydata.reader:
                if row['ASSAY_NAME'] != mydata.assay:
                    continue

                try:
                    selfie = sf.encoder(row['SMILES'])
                except sf.EncoderError:
                    selfie = nan
                mydata.descriptor_results = [row['PREFERRED_NAME'], row['SMILES'], selfie, row['HIT_CALL']]

                mol = Chem.MolFromSmiles(row['SMILES'])
                for method in mydata.descriptor_methods:
                    mydata.descriptor_results.append(method(mol))
                
                mydata.writer.writerow(mydata.descriptor_results)
                count += 1
        print("Assay " + mydata.assay + " complete", flush = True)
        return "Assay " + mydata.assay + " complete"

#Just used this to test if multhreading was working properly
def printsomething(wrapper):
    mydata = threading.local()
    mydata.assay, mydata.inputfile, mydata.outputfile = wrapper
    print(mydata.assay, mydata.inputfile, mydata.outputfile)

                    
#You'll need to download the two .csv read files from Sohaib's folder on the shared drive, they're too large for github
def run_program():
    assay_name_file = "./Data/toxcast_active.csv"
    with open(assay_name_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)

        assay_list =  ["TOX21_Aromatase_Inhibition", "TOX21_p53_BLA_p1_ratio", "TOX21_ARE_BLA_agonist_ratio", "TOX21_NFkB_BLA_agonist_ratio", "TOX21_AhR_LUC_Agonist", "TOX21_AR_BLA_Agonist_ratio", "TOX21_CAR_Agonist", 
            "TOX21_ERa_BLA_Agonist_ratio", "TOX21_ERb_BLA_Agonist_ratio", "TOX21_ERR_Agonist", "TOX21_AhR_LUC_Agonist_viability", "TOX21_AR_LUC_MDAKB2_Antagonist_10nM_R1881_viability", "TOX21_p53_BLA_p1_viability"]
        count = 0
        # for row in reader:
        #     if row['ASSAY_NAME'] not in assay_list:
        #         assay_list.append(row['ASSAY_NAME'])

        args = []
        for assay in assay_list:
            count += 1
            args.append((assay, "./Data/toxcast_large.csv", "./HarnessData/" + assay + ".csv"))
            
        print(args)
        
        pool = Pool(os.cpu_count() - 1)
        pool.map(createDBByName, args)

        print("db complete")

if __name__ == '__main__':


    run_program()