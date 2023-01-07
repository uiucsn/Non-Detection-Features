import pandas as pd
import os
from multiprocessing import Pool
from astropy.coordinates import SkyCoord, ICRS
from astropy import units as u
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.pyplot import cla
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

def plotGenricSkyMap(coords):
    """
    A generic function to plot a skymap for the given Sky coord array.

    Args:
        coords (numpy array): A numpy array of skycoord objects
    """
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="mollweide")
    scatter = ax.scatter(-coords.ra.wrap_at(180 * u.deg).radian, coords.dec.wrap_at(180 * u.deg).radian, s=3, vmin=0)
    ax.grid(True)
    ax.set_xticklabels(['10h', '8h', '6h', '4h', '2h', '0h', '22h', '20h', '18h', '16h', '14h'])
    plt.show()

def getEncodedData(df, enc):

    # Removing duplicate SNID. Happens because flares can be in the sky maps for multiple KN's
    df = df.drop_duplicates(subset=['SNID'])

    # Make the matrices
    x = df[['BAND','PRE-BAND','POST-BAND']]
    x = enc.transform(x)

    # Adding the time to prev and next det as features
    x_new = np.zeros((len(df), 24))
    x_new[:, :20] = x
    x_new[:, 20] = df['TIME-TO-PREV']
    x_new[:, 21] = df['TIME-TO-NEXT']
    x_new[:, 22] = df['MDF_DENSITY']
    x_new[:, 23] = df['GW_PROB']
    # x_new[:, 23] = df['NEXT-PHOT-FLAG']
    # x_new[:, 24] = df['NUM_DETECTIONS']


    features = list(enc.get_feature_names_out())
    features.append('TIME-TO-PREV')
    features.append('TIME-TO-NEXT')
    features.append('MDF_DENSITY')
    features.append('GW_PROB')
    # features.append('NEXT-PHOT-FLAG')
    # features.append('NUM_DETECTIONS')

    # Creating binary class column 
    y_new = []
    for c in df['CLASS']:
        if c == 'KN Bulla' or c == 'KN Kasen':
            y_new.append(0)
        elif c == 'MDF':
            y_new.append(1)

    x_new = pd.DataFrame(x_new, columns=features)
    y_new = pd.DataFrame(y_new, columns=['CLASS'])
    return x_new, y_new

mappings_bulla = pd.read_csv('test_data/MDF_VS_KN-KilonovaSims/Bulla/SNID_TO_SKYMAP.csv')
mappings_kasen = pd.read_csv('test_data/MDF_VS_KN-KilonovaSims/Kasen/SNID_TO_SKYMAP.csv')

bulla_features = []
kasen_features = []

for i in range(len(mappings_bulla)):

    SNID = mappings_bulla['SNID'][i]

    try:
        path = f'Bulla_features/{SNID}.csv'
        df = pd.read_csv(path)
        bulla_features.append(df)
    except:
        print(f'Bulla SNID {SNID} features not found')

for i in range(len(mappings_kasen)):

    SNID = mappings_kasen['SNID'][i]

    try:
        path = f'Kasen_features/{SNID}.csv'
        df = pd.read_csv(path)
        kasen_features.append(df)
    except:
        print(f'Kasen SNID {SNID} features not found. ')

train_fraction = 0.6

bulla_train_size = int(train_fraction * len(bulla_features))
bulla_train = pd.concat(bulla_features[:bulla_train_size])
bulla_test = pd.concat(bulla_features[bulla_train_size:])

kasen_train_size = int(train_fraction * len(kasen_features))
kasen_train = pd.concat(kasen_features[:kasen_train_size])
kasen_test = pd.concat(kasen_features[kasen_train_size:])


complete_training_set = pd.concat([kasen_train, bulla_train])

# Removing duplicate SNID. Happens because flares can be in the sky maps for multiple KN's
complete_training_set = complete_training_set.drop_duplicates(subset=['SNID'])


all_SNID = complete_training_set['SNID']

# SNID of m dwarf flares only
mdf_SNID = complete_training_set[complete_training_set['CLASS'] == 'MDF']['SNID']

# Use the last 20 % of flare SNID just for validating
test_SNID = mdf_SNID[int(0.8 * len(mdf_SNID)):]

# Remove any SNID's used for testing from the training set
complete_training_set = complete_training_set[complete_training_set.SNID.isin(test_SNID) == False]

train_SNID = complete_training_set['SNID']

# Make the matrices
x = complete_training_set[['BAND','PRE-BAND','POST-BAND']]

# One hot encoding for passband features
enc = OneHotEncoder(sparse=False)

enc.fit(x)

#Classifier
x_train, y_train = getEncodedData(complete_training_set, enc)

weights = {
    0:0.95, # KN
    1:0.05  # MDF
}


print('Fitting model')
clf=RandomForestClassifier(n_estimators=1000, random_state=42, class_weight=weights)
clf.fit(x_train, y_train['CLASS'])

n_max = 50

bulla_fractions = []
kasen_fractions = []
n_values = []


for n in range(1, n_max + 1):

    total_bulla = 0
    found_kn_bulla = 0

    for table in bulla_features[bulla_train_size:]:

        table = table[table.SNID.isin(train_SNID) == False]

        test_x, test_y = getEncodedData(table, enc)
        probs_kn = clf.predict_proba(test_x)[:, 1]

        # Sort by prob of being a KN and get the first 5 indices
        idx = np.argsort(probs_kn)[:min(n, len(probs_kn))]
        sorted_probs = probs_kn[idx]

        y_true = test_y['CLASS'][idx]

        # Check if the KN exists in the top n candidates
        if 0 in y_true:
            found_kn_bulla += 1

        total_bulla += 1


    print(f'Found {found_kn_bulla} out of {total_bulla} kilo nova when n = {n}.')


    total_kasen = 0
    found_kn_kasen = 0

    for table in kasen_features[kasen_train_size:]:

        table = table[table.SNID.isin(train_SNID) == False]

        test_x, test_y = getEncodedData(table, enc)
        probs_kn = clf.predict_proba(test_x)[:, 1]

        # Sort by prob of being a KN and get the first 5 indices
        idx = np.argsort(probs_kn)[:min(n, len(probs_kn))]
        sorted_probs = probs_kn[idx]

        y_true = test_y['CLASS'][idx]

        # Check if the KN exists in the top n candidates
        if 0 in y_true:
            found_kn_kasen += 1

        total_kasen += 1

    print(f'Found {found_kn_kasen} out of {total_kasen} kilo nova when n = {n}.')

    n_values.append(n)
    bulla_fractions.append(found_kn_bulla/total_bulla)
    kasen_fractions.append(found_kn_kasen/total_kasen)


plt.plot(n_values, bulla_fractions, label='Bulla')
plt.plot(n_values, kasen_fractions, label='Kasen')
plt.ylabel('Fraction of KN found')
plt.xlabel('Maximum number of candidates considered')
plt.legend()
plt.show()