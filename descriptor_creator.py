import pandas as pd
import numpy as np

#A variety of balanced AND large adtasets. A successful model would score better than activity ratio.

#60.1% active: ACEA_AR_antagonist_80hr (1770 sample size)
#51.6%: ATG_PXRE_CIS_up (3522)
#57.8%: BSK_hDFCGF_Proliferation_down (1430)
#52.8%: LTEA_HepaRG_CYP1A2_up (1016)
#47.6%: LTEA_HepaRG_CYP2B6_up (1016)
#52.4%: LTEA_HepaRG_CYP2E1_dn (1016)
#53.3%: LTEA_HepaRG_SLC10A1_dn (1016)
#61.9%: TOX21_DT40 (7948)
#59.6%: TOX21_DT40_100 (7948)
#62.9%: TOX21_DT40_657 (7948)

#All you need to do is change this list to generate new pkl files, provided you have the correct files downloaded
assays = ["ACEA_AR_antagonist_80hr", "ATG_PXRE_CIS_up", "BSK_hDFCGF_Proliferation_down", "LTEA_HepaRG_CYP1A2_up", "LTEA_HepaRG_CYP2B6_up", "LTEA_HepaRG_CYP2E1_dn", "LTEA_HepaRG_SLC10A1_dn", "TOX21_DT40", "TOX21_DT40_100", "TOX21_DT40_657"]
#

#To change the descriptors, change the next two lines accordingly.
mordred = df = pd.read_pickle("./Reisfeld_pkls/mordred.pkl").values
#For mordred, SMILES is at last index for some reason
trash, mordred, m_smiles = np.split(mordred, [1, 1614], axis = 1)

for assay in assays:
    assay_df = pd.read_table('./Data/' + assay + '.csv', sep=',')

    #Output df will have the format: 
    #[SMILES, ACTIVITY, [descriptors]]
    assay_smiles = assay_df['SMILES'].values
    activity = assay_df['ACTIVE'].values

    #intersection between mordred indicators
    inter = np.intersect1d(m_smiles, assay_smiles)
    m_dict = {}

    export_array = []
    for overlap in inter:
        m_dict[overlap] = mordred[np.where(m_smiles == overlap)[0],:]
        if m_dict[overlap].size == 0:
            continue

    for smile in inter:
        #below is guaranteed
        index = np.where(assay_smiles == smile)[0]
        row = np.array([[smile, activity[index][0]]])

        row = np.concatenate((row, [m_dict[smile][0]]), axis = 1)
        
        if len(export_array) == 0:
            export_array = row
        else:
            #this is the main issue with efficiency/speed   
            export_array = np.concatenate((export_array, row), axis = 0)     

    pd.DataFrame(export_array).to_pickle(path = "./TestData/" + assay + "_activity.pkl", compression='infer', protocol=5, storage_options=None)
    print("Assay: ", assay, " done")