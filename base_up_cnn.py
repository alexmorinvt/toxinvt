import pandas as pd
import numpy as np
import tensorflow as tf

class LearningRateReducerCb(tf.keras.callbacks.Callback):

  def on_epoch_end(self, epoch, logs={}):
    old_lr = self.model.optimizer.lr.read_value()
    new_lr = old_lr * 0.999
    print("\nEpoch: {}. Reducing Learning Rate from {} to {}".format(epoch, old_lr, new_lr))
    self.model.optimizer.lr.assign(new_lr)

def leaky_relu(z):
    return np.maximum(0.01 * z, z)

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
assays = ["ACEA_AR_antagonist_80hr"]#, "ATG_PXRE_CIS_up", "BSK_hDFCGF_Proliferation_down", "LTEA_HepaRG_CYP1A2_up", "LTEA_HepaRG_CYP2B6_up", "LTEA_HepaRG_CYP2E1_dn", "LTEA_HepaRG_SLC10A1_dn", "TOX21_DT40", "TOX21_DT40_100", "TOX21_DT40_657"]
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

    x = 0
    for smile in inter:
        if x > 100:
            break
        #below is guaranteed
        index = np.where(assay_smiles == smile)[0]
        row = np.array([[smile, activity[index][0]]])

        row = np.concatenate((row, [m_dict[smile][0][:10]]), axis = 1)
        
        if len(export_array) == 0:
            export_array = row
        else:
            #this is the main issue with efficiency/speed   
            export_array = np.concatenate((export_array, row), axis = 0)   
        x+=1  
    print(export_array.shape)

assay_data = export_array

#axis = 1 signifies a vertical split (column wise). Currently set at 90%
train, test = np.split(assay_data, [int(0.9 * len(assay_data))], axis = 0)

train_smiles, train_activity, train_descriptors = np.split(train, [1, 2], axis = 1)
test_smiles, test_activity, test_descriptors = np.split(test, [1, 2], axis = 1)

#Processing np array - changing to float, normalizing, and removing nan
train_activity = np.asarray(train_activity).astype('float64')
train_descriptors = np.asarray(train_descriptors).astype('float64')
test_activity = np.asarray(test_activity).astype('float64')
test_descriptors = np.asarray(test_descriptors).astype('float64')

train_activity = np.nan_to_num(train_activity).astype(int)
train_descriptors = np.nan_to_num(train_descriptors)
test_activity = np.nan_to_num(test_activity).astype(int)
test_descriptors = np.nan_to_num(test_descriptors)

train_descriptors = np.add(train_descriptors, 0.0000000001)
test_descriptors = np.add(test_descriptors, 0.0000000001)

train_descriptors = train_descriptors / (train_descriptors.max(axis=0) + 0.00000001)
test_descriptors = test_descriptors / (test_descriptors.max(axis=0) + 0.00000001)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(208, activation='leaky_relu'),
    tf.keras.layers.Dense(208, activation='leaky_relu'),
    tf.keras.layers.Dense(208, activation='leaky_relu'),
    tf.keras.layers.Dense(208, activation='leaky_relu'),
    tf.keras.layers.Dense(208, activation='leaky_relu'),
    tf.keras.layers.Dense(208, activation='leaky_relu'),
    tf.keras.layers.Dense(208, activation='leaky_relu'),
    tf.keras.layers.Dense(1, activation = 'sigmoid')
])

#standard values are lr = 0.001, b1 = 0.9, b2 = 0.999., ep = 1e-07, amsgrad = False
opt = tf.keras.optimizers.Adam(
    learning_rate=0.09, beta_1=0.9, beta_2=0.999, epsilon=0.1, amsgrad=False,
    name='Adam',
)

opt = tf.keras.optimizers.SGD(
    learning_rate=0.01, momentum=0.0, nesterov=False, name="SGD"
)

class_weight = {0: 1.,
            1: 1.}

model.compile(optimizer=opt,
            loss='binary_crossentropy',
            metrics=['accuracy'])

model.fit(train_descriptors, train_activity, callbacks=[LearningRateReducerCb()], epochs=5000, class_weight = class_weight)

model.evaluate(test_descriptors, test_activity, verbose=2)
predictions = model.predict(test_descriptors)
predictions = (predictions[:,0] > 0.5).astype(np.int).ravel()
test_activity = test_activity.ravel()

print(tf.math.confusion_matrix(tf.convert_to_tensor(predictions), test_activity))