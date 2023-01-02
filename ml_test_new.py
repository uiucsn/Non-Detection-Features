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

mappings_bulla = pd.read_csv('test_data/MDF_VS_KN-KilonovaSims/Bulla/SNID_TO_SKYMAP.csv')
mappings_kasen = pd.read_csv('test_data/MDF_VS_KN-KilonovaSims/Kasen/SNID_TO_SKYMAP.csv')

l = []

for i in range(len(mappings_bulla)):

    SNID = mappings_bulla['SNID'][i]

    try:
        path = f'Bulla_features/{SNID}.csv'
        df = pd.read_csv(path)
        l.append(df)
    except:
        print(f'Bulla SNID {SNID} not found')

for i in range(len(mappings_kasen)):

    SNID = mappings_kasen['SNID'][i]

    try:
        path = f'Kasen_features/{SNID}.csv'
        df = pd.read_csv(path)
        l.append(df)
    except:
        print(f'Kasen SNID {SNID} not found')

df = pd.concat(l)

# Removing duplicate SNID. Happens because flares can be in the sky maps for multiple KN's
df = df.drop_duplicates(subset=['SNID'])

# c = SkyCoord(ra=df['RA'], dec=df['DEC'], frame=ICRS, unit='deg')
# plotGenricSkyMap(c)

# flares
df_1 = df[df['CLASS']=='MDF']

# KN
df_Kasen = df[df['CLASS']=='KN Kasen']
df_Bulla = df[df['CLASS']=='KN Bulla']
print(len(df_Bulla), len(df_Kasen))
df_2 = pd.concat([df_Bulla, df_Kasen])


df_1 = df_1[:len(df_2)]


# Merge the dataframs
df = pd.concat([df_1,df_2])

# Make the matrices
x = df[['BAND','PRE-BAND','POST-BAND']]
y = df[['CLASS']]

# One hot encoding for passband features
enc = OneHotEncoder(sparse=False)
x = enc.fit_transform(x)

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

x_new = pd.DataFrame(x_new, columns=features)

x = x_new

print(x_new)

# Creating binary class column 
y = []
for c in df['CLASS']:
    if c == 'KN Bulla' or c == 'KN Kasen':
        y.append(0)
    else:
        y.append(1)

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)

# Classifier
clf=RandomForestClassifier(n_estimators=1000, random_state=42)

clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)



print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)

importance = clf.feature_importances_

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Kilo Nova', 'M Dwarf flare'])
disp.plot()
plt.show()

for i in range(len(importance)):
    print(f'{features[i]}: {importance[i]}')