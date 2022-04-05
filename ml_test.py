from xml.etree.ElementInclude import include
from matplotlib.pyplot import axis, cla
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

# Load data
df_1 = pd.read_csv('KN_features.csv')
df_2 = pd.read_csv('MDF_features.csv')

# Make the number of detections for the two classes equal
df_2 = df_2[:len(df_1)]

# Merge the dataframs
df = pd.concat([df_1,df_2])

print(df)

# Make the matrices
x = df.drop(['CLASS'], axis=1)
y = df[['CLASS']]

# One hot encoding for passband features
enc = OneHotEncoder(sparse=False)
x_enc = enc.fit_transform(x[['BAND', 'PRE-BAND','POST-BAND']])

# Getting encodded feature list
features = list(enc.get_feature_names_out())

# Creating the data frame for the encoded data
x_enc = pd.DataFrame(x_enc, columns=features)

# Dropping the encodded data
x = x.drop(['BAND', 'PRE-BAND','POST-BAND'], axis=1)

# Joining the encoded data with the other features
x = x_enc.merge(x, left_index=True, right_index=True)


# Creating binary class column 
y = []
for c in df['CLASS']:
    if c == 'KN':
        y.append(0)
    else:
        y.append(1)


# Splitting data
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# Classifier
clf=RandomForestClassifier(n_estimators=1000)

clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)



print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)

importance = clf.feature_importances_

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot()
plt.show()

for i in range(len(importance)):
    print(f'{features[i]}: {importance[i]}')


