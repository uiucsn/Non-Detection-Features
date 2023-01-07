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

full_true_bulla = np.array([])
full_pred_bulla = np.array([])

total_bulla = 0
found_kn_bulla = 0

for table in bulla_features[bulla_train_size:]:

    table = table[table.SNID.isin(train_SNID) == False]

    test_x, test_y = getEncodedData(table, enc)
    probs = clf.predict_proba(test_x)

    y_pred = clf.predict(test_x)

    if y_pred[0] == 0 and test_y['CLASS'][0] == 0:
        found_kn_bulla += 1


    #print("Accuracy:",metrics.accuracy_score(test_y, y_pred))
    cm = confusion_matrix(test_y['CLASS'], y_pred, labels=clf.classes_)

    importance = clf.feature_importances_

    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Kilo Nova', 'M Dwarf flare'])
    # disp.plot()
    # SNID = table['SNID'][0]
    #plt.savefig(f'find_one_kn/{SNID}.png')

    full_true_bulla = np.append(full_true_bulla, test_y['CLASS'])
    full_pred_bulla = np.append(full_pred_bulla, y_pred)

    total_bulla += 1

#print("Accuracy:",metrics.accuracy_score(test_y, y_pred))
cm = confusion_matrix(full_true_bulla, full_pred_bulla, labels=clf.classes_)

importance = clf.feature_importances_

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Kilo Nova', 'M Dwarf flare'])
disp.plot()
plt.title('Testing for Bulla simulations')
plt.show()

print(f'Found {found_kn_bulla} out of {total_bulla} kilo nova.')




full_true_kasen = np.array([])
full_pred_kasen = np.array([])

total_kasen = 0
found_kn_kasen = 0

for table in kasen_features[kasen_train_size:]:

    table = table[table.SNID.isin(train_SNID) == False]

    test_x, test_y = getEncodedData(table, enc)
    probs = clf.predict_proba(test_x)

    y_pred = clf.predict(test_x)

    if y_pred[0] == 0 and test_y['CLASS'][0] == 0:
        found_kn_kasen += 1


    #print("Accuracy:",metrics.accuracy_score(test_y, y_pred))
    cm = confusion_matrix(test_y['CLASS'], y_pred, labels=clf.classes_)

    importance = clf.feature_importances_

    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Kilo Nova', 'M Dwarf flare'])
    # disp.plot()
    # SNID = table['SNID'][0]
    #plt.savefig(f'find_one_kn/{SNID}.png')

    full_true_kasen = np.append(full_true_kasen, test_y['CLASS'])
    full_pred_kasen = np.append(full_pred_kasen, y_pred)

    total_kasen += 1

#print("Accuracy:",metrics.accuracy_score(test_y, y_pred))
cm = confusion_matrix(full_true_kasen, full_pred_kasen, labels=clf.classes_)

importance = clf.feature_importances_

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Kilo Nova', 'M Dwarf flare'])
disp.plot()
plt.title('Testing for Kasen simulations')
plt.show()

print(f'Found {found_kn_kasen} out of {total_kasen} kilo nova.')



# found_Bulla = 0 # Number of entries correctly marked as KN
# true_Bulla = 0 # Number of KN's
# marked_Bulla = 0 # Number of entries marked as KN

# for i in range(int(0.8 * len(mappings_bulla)), len(mappings_bulla)):

#     SNID = mappings_bulla['SNID'][i]

#     try:
#         path = f'Bulla_features/{SNID}.csv'
#         df = pd.read_csv(path)
#         df = df[:min(100, len(df))]
        
#         # Make the matrices
#         x = df[['BAND','PRE-BAND','POST-BAND']]
#         y = df[['CLASS']]

#         # One hot encoding for passband features
#         enc = OneHotEncoder(sparse=False)
#         x = enc.fit_transform(x)

#         # Adding the time to prev and next det as features
#         x_new = np.zeros((len(df), 24))
#         x_new[:, :20] = x
#         x_new[:, 20] = df['TIME-TO-PREV']
#         x_new[:, 21] = df['TIME-TO-NEXT']
#         x_new[:, 22] = df['MDF_DENSITY']
#         x_new[:, 23] = df['GW_PROB']
#         # x_new[:, 23] = df['NEXT-PHOT-FLAG']
#         # x_new[:, 24] = df['NUM_DETECTIONS']


#         features = list(enc.get_feature_names_out())
#         features.append('TIME-TO-PREV')
#         features.append('TIME-TO-NEXT')
#         features.append('MDF_DENSITY')
#         features.append('GW_PROB')
#         # features.append('NEXT-PHOT-FLAG')
#         # features.append('NUM_DETECTIONS')

#         x_new = pd.DataFrame(x_new, columns=features)

#         x = x_new

#         # Creating binary class column 
#         y = []
#         for c in df['CLASS']:
#             if c == 'KN Bulla' or c == 'KN Kasen':
#                 y.append(0)
#             else:
#                 y.append(1)

#         y_pred=clf.predict(x_new)
#         if y_pred[0] == y[0]:
#             found_Bulla += 1
#         true_Bulla += 1
#         marked_Bulla += len(y_pred == 0)
#         print("Accuracy:",metrics.accuracy_score(y, y_pred))
#         cm = confusion_matrix(y, y_pred, labels=clf.classes_)

#         importance = clf.feature_importances_

#         disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Kilo Nova', 'M Dwarf flare'])
#         disp.plot()
#         #plt.show()

#     except:
#         m = 0

# print(f'Found {found_Bulla} of {true_Bulla} KN and incorrectly marked {marked_Bulla - true_Bulla}')


# found_Bulla = 0 # Number of entries correctly marked as KN
# true_Bulla = 0 # Number of KN's
# marked_Bulla = 0 # Number of entries marked as KN

# for i in range(int(0.8 * len(mappings_kasen)), len(mappings_kasen)):

#     SNID = mappings_bulla['SNID'][i]

#     try:
#         path = f'Kasen_features/{SNID}.csv'
#         df = pd.read_csv(path)
#         print(path)
#         df = df[:min(100, len(df))]
        
#         # Make the matrices
#         x = df[['BAND','PRE-BAND','POST-BAND']]
#         y = df[['CLASS']]

#         # One hot encoding for passband features
#         enc = OneHotEncoder(sparse=False)
#         x = enc.fit_transform(x)

#         # Adding the time to prev and next det as features
#         x_new = np.zeros((len(df), 24))
#         x_new[:, :20] = x
#         x_new[:, 20] = df['TIME-TO-PREV']
#         x_new[:, 21] = df['TIME-TO-NEXT']
#         x_new[:, 22] = df['MDF_DENSITY']
#         x_new[:, 23] = df['GW_PROB']
#         # x_new[:, 23] = df['NEXT-PHOT-FLAG']
#         # x_new[:, 24] = df['NUM_DETECTIONS']


#         features = list(enc.get_feature_names_out())
#         features.append('TIME-TO-PREV')
#         features.append('TIME-TO-NEXT')
#         features.append('MDF_DENSITY')
#         features.append('GW_PROB')
#         # features.append('NEXT-PHOT-FLAG')
#         # features.append('NUM_DETECTIONS')

#         x_new = pd.DataFrame(x_new, columns=features)

#         x = x_new

#         # Creating binary class column 
#         y = []
#         for c in df['CLASS']:
#             if c == 'KN Bulla' or c == 'KN Kasen':
#                 y.append(0)
#             else:
#                 y.append(1)

#         y_pred=clf.predict(x_new)
#         if y_pred[0] == y[0]:
#             found_Bulla += 1
#         true_Bulla += 1
#         marked_Bulla += len(y_pred == 0)
#         print("Accuracy:",metrics.accuracy_score(y, y_pred))
#         cm = confusion_matrix(y, y_pred, labels=clf.classes_)

#         importance = clf.feature_importances_

#         disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Kilo Nova', 'M Dwarf flare'])
#         disp.plot()
#         plt.show()

#     except:
#         m = 0
        
# print(f'Found {found_Bulla} of {true_Bulla} KN and incorrectly marked {marked_Bulla - true_Bulla}')



