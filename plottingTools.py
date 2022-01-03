from datetime import time
from numpy.core.defchararray import less_equal
from numpy.core.fromnumeric import size
import sncosmo
import matplotlib.pyplot as plt
import numpy as np
import NDFeatures
from collections import Counter
import pandas as pd
import sklearn as sk
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import collections


def plotDetectionPassbands(path, listFile, isCompressed):
    
    detectionPassband = np.array([])

    with open(path + listFile) as file:
        for line in file:

            headFile = path + line.rstrip()
            photFile = headFile[:len(headFile) - 9] + 'PHOT.FITS'

            if isCompressed:
                headFile += '.gz'
                photFile += '.gz'

            sim = sncosmo.read_snana_fits(headFile, photFile)

            for table in sim:

                fe = NDFeatures.NDFeatureExtractor(table, 'LSST')

                detectionData = fe.extractDetectionData()
                detectionPassband = np.concatenate([detectionPassband, detectionData['BAND'].to_numpy()])
        
    # Plotting code
    detectionPassband = sorted(detectionPassband)
                
    letter_counts = Counter(detectionPassband)
    df = pd.DataFrame.from_dict(letter_counts, orient='index')
    df = df.reindex(['u ','g ','r ','i ','z ','Y '])
    df.plot(kind='bar')

    plt.title('Distribution of first detection passbands ' + dirToClassName[path])
    plt.show()

def plotPreDetectionPassbands(path, listFile, isCompressed):

    preDetectionPassband = np.array([])

    with open(path + listFile) as file:
        for line in file:

            headFile = path + line.rstrip()
            photFile = headFile[:len(headFile) - 9] + 'PHOT.FITS'

            if isCompressed:
                headFile += '.gz'
                photFile += '.gz'

            sim = sncosmo.read_snana_fits(headFile, photFile)

            for table in sim:

                fe = NDFeatures.NDFeatureExtractor(table, 'LSST')
                preDetectionData = fe.extractPrecceedingObservations()

                for df in preDetectionData:
                    preDetectionPassband = np.concatenate([preDetectionPassband, df['BAND'].to_numpy()])

    # Plotting code
    preDetectionPassband = sorted(preDetectionPassband)

    letter_counts = Counter(preDetectionPassband)
    df = pd.DataFrame.from_dict(letter_counts, orient='index')
    df = df.reindex(['u ','g ','r ','i ','z ','Y '])
    df.plot(kind='bar')

    plt.title('Distribution of pre detection passbands ' + dirToClassName[path])
    plt.show()

def plotPostDetectionPassbands(path, listFile, isCompressed):
    
    postDetectionPassbands = np.array([])

    with open(path + listFile) as file:
        for line in file:

            headFile = path + line.rstrip()
            photFile = headFile[:len(headFile) - 9] + 'PHOT.FITS'

            if isCompressed:
                headFile += '.gz'
                photFile += '.gz'

            sim = sncosmo.read_snana_fits(headFile, photFile)

            for table in sim:

                fe = NDFeatures.NDFeatureExtractor(table, 'LSST')
                postDetectionData = fe.extractSucceedingObservations()

                for df in postDetectionData:
                    postDetectionPassbands = np.concatenate([postDetectionPassbands, df['BAND'].to_numpy()])   

    # Plotting code
    postDetectionPassbands = sorted(postDetectionPassbands)

    letter_counts = Counter(postDetectionPassbands)
    df = pd.DataFrame.from_dict(letter_counts, orient='index')
    df = df.reindex(['u ','g ','r ','i ','z ','Y '])
    df.plot(kind='bar')

    plt.title('Distribution of post detection passbands ' + dirToClassName[path])
    plt.show()


def plotSuccessiveDetectionTimeDelta(path, listFile, isCompressed):
    
    timeDelta = []

    with open(path + listFile) as file:
        for line in file:

            headFile = path + line.rstrip()
            photFile = headFile[:len(headFile) - 9] + 'PHOT.FITS'

            if isCompressed:
                headFile += '.gz'
                photFile += '.gz'

            sim = sncosmo.read_snana_fits(headFile, photFile)

            for table in sim:

                fe = NDFeatures.NDFeatureExtractor(table, 'LSST')

                timeBetweenDetections = fe.extractTimeBetweenSucessiveDetections()
                timeDelta += timeBetweenDetections['Time Delta']

    # Plotting code
    plt.hist(timeDelta)
    plt.title('Distribution of post detection passbands ' + dirToClassName[path])
    plt.show()

def plotSignalToNoiseRatio(path, listFile, isCompressed):
    
    signalToNoiseRatio = np.array([])

    with open(path + listFile) as file:
        count = 0
        for line in file:

            headFile = path + line.rstrip()
            photFile = headFile[:len(headFile) - 9] + 'PHOT.FITS'

            if isCompressed:
                headFile += '.gz'
                photFile += '.gz'
            sim = sncosmo.read_snana_fits(headFile, photFile)

            for table in sim:

                fe = NDFeatures.NDFeatureExtractor(table, 'LSST')
                signalToNoiseDF = fe.extractSignalToNoiseRatio()
                
                fe.plotPseudoLightCurves()
                signalToNoiseRatio = np.concatenate([signalToNoiseRatio, signalToNoiseDF['Signal to noise'].to_numpy()[0]])

    # Plotting code
    plt.hist(signalToNoiseRatio, bins=100)
    plt.yscale('log')
    plt.title('Distribution of signal to noise ratio ' + dirToClassName[path])
    plt.show()

def plotPreDetectionPassbandConfusionMatrix(path, listFile, isCompressed):

    detectionPassband = np.array([])
    preDetectionPassband = np.array([])

    with open(path + listFile) as file:
        for line in file:

            headFile = path + line.rstrip()
            photFile = headFile[:len(headFile) - 9] + 'PHOT.FITS'

            if isCompressed:
                headFile += '.gz'
                photFile += '.gz'

            sim = sncosmo.read_snana_fits(headFile, photFile)

            for table in sim:

                fe = NDFeatures.NDFeatureExtractor(table, 'LSST')

                detectionData = fe.extractDetectionData()
                preDetectionData = fe.extractPrecceedingObservations()

                count = 0
                for df in preDetectionData:
                    if len(df) >= 1:
                        preDetectionPassband = np.concatenate([preDetectionPassband, df['BAND'].to_numpy()])
                        detectionPassband = np.append(detectionPassband, detectionData['BAND'].to_numpy()[count])
                    count += 1
    
    # Plotting code
    cm = confusion_matrix(detectionPassband, preDetectionPassband, labels=['u ','g ','r ','i ','z ','Y '])

    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=['u ','g ','r ','i ','z ','Y '])
    disp.plot()

    plt.title('Preceeding non detection passband correlation with detection passband ' + dirToClassName[path])
    plt.ylabel('Detection passband')
    plt.xlabel('Pre detection passband')
    plt.show()

def plotPostDetectionPassbandConfusionMatrix(path, listFile, isCompressed):

    detectionPassband = np.array([])
    postDetectionPassband = np.array([])

    with open(path + listFile) as file:
        for line in file:

            headFile = path + line.rstrip()
            photFile = headFile[:len(headFile) - 9] + 'PHOT.FITS'

            if isCompressed:
                headFile += '.gz'
                photFile += '.gz'

            sim = sncosmo.read_snana_fits(headFile, photFile)

            for table in sim:

                fe = NDFeatures.NDFeatureExtractor(table, 'LSST')

                detectionData = fe.extractDetectionData()
                postDetectionData = fe.extractSucceedingObservations()

                count = 0
                for df in postDetectionData:
                    if len(df) >= 1:
                        postDetectionPassband = np.concatenate([postDetectionPassband, df['BAND'].to_numpy()])
                        detectionPassband = np.append(detectionPassband, detectionData['BAND'].to_numpy()[count])
                    count += 1
    
    # Plotting code
    cm = confusion_matrix(detectionPassband, postDetectionPassband, labels=['u ','g ','r ','i ','z ','Y '])

    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=['u ','g ','r ','i ','z ','Y '])
    disp.plot()

    plt.title('Suceeding non detection passband correlation with detection passband ' + dirToClassName[path])
    plt.ylabel('Detection passband')
    plt.xlabel('Post detection passband')
    plt.show()


def getPrePostAndDetectionPassbands(path, listFile, isCompressed):

    detectionPassband = np.array([])
    postDetectionPassband = np.array([])
    preDetectionPassband = np.array([])

    with open(path + listFile) as file:
        for line in file:

            headFile = path + line.rstrip()
            photFile = headFile[:len(headFile) - 9] + 'PHOT.FITS'

            if isCompressed:
                headFile += '.gz'
                photFile += '.gz'

            sim = sncosmo.read_snana_fits(headFile, photFile)

            for table in sim:

                fe = NDFeatures.NDFeatureExtractor(table, 'LSST')

                detectionData = fe.extractDetectionData()
                preDetectionData = fe.extractPrecceedingObservations()
                postDetectionData = fe.extractSucceedingObservations()

                count = 0
                for df1, df2 in zip(preDetectionData,postDetectionData):
                    if len(df1) == 1 and len(df2) == 1:
                        preDetectionPassband = np.concatenate([preDetectionPassband, df1['BAND'].to_numpy()])
                        postDetectionPassband = np.concatenate([postDetectionPassband, df2['BAND'].to_numpy()])
                        detectionPassband = np.append(detectionPassband, detectionData['BAND'].to_numpy()[count])
                    count += 1
    return preDetectionPassband, detectionPassband, postDetectionPassband

def plotAllObservationPassbandDistribution(path, listFile, isCompressed):

    passbands = np.array([])

    with open(path + listFile) as file:
        for line in file:

            headFile = path + line.rstrip()
            photFile = headFile[:len(headFile) - 9] + 'PHOT.FITS'

            if isCompressed:
                headFile += '.gz'
                photFile += '.gz'

            sim = sncosmo.read_snana_fits(headFile, photFile)

            for table in sim:
                temp = np.array(table['BAND'], dtype=np.str)
                passbands = np.concatenate([passbands, temp])   

    # Plotting code 
    passbands = sorted(passbands)

    letter_counts = Counter(passbands)
    df = pd.DataFrame.from_dict(letter_counts, orient='index')
    df = df.reindex(['u ','g ','r ','i ','z ','Y '])
    df.plot(kind='bar')

    plt.title('Distribution of non detection and detection passbands ' + dirToClassName[path])
    plt.show()

def convertToNumbers(passbandArray):
    toReturn = []
    for band in passbandArray:
        if band == 'u ':
            toReturn.append(0)
        elif band == 'g ':
            toReturn.append(1)
        elif band == 'r ':
            toReturn.append(2)
        elif band == 'i ':
            toReturn.append(3)
        elif band == 'z ':
            toReturn.append(4)
        elif band == 'Y ':
            toReturn.append(5)
    return toReturn
    
def plotPrePost3dPlot():

    # Kilo nova data
    KnPre, KnDet, KnPost = getPrePostAndDetectionPassbands('test_data/kasen-kilonova-lightcurves/DC_LSST_MODEL_KN17_WITH_HOST_EXT/', 'DC_LSST_MODEL_KN17_WITH_HOST_EXT.LIST', False)
    KnPre = convertToNumbers(KnPre)
    KnDet = convertToNumbers(KnDet)
    KnPost = convertToNumbers(KnPost)

    KnTuples = []
    for a, b, c in zip(KnPre, KnDet, KnPost):
        KnTuples.append((a,b,c))

    KnOccurrences = collections.Counter(KnTuples)
    KnCount = sum(KnOccurrences.values())

    # M dwarf flare data
    MdPre, MdDet, MdPost = getPrePostAndDetectionPassbands('test_data/m-dwarf-flare-lightcurves/','LSST_WFD_MODEL66_Mdwarf.LIST', True)
    MdPre = convertToNumbers(MdPre)
    MdDet = convertToNumbers(MdDet)
    MdPost = convertToNumbers(MdPost)

    MdTuples = []
    for a, b, c in zip(MdPre, MdDet, MdPost):
        MdTuples.append((a,b,c))

    MdOccurrences = collections.Counter(MdTuples)
    MdCount = sum(MdOccurrences.values())

    # Plotting code

    fig = plt.figure(figsize = (10, 7))
    ax = plt.axes(projection ="3d")

    
    for key in KnOccurrences:
        ax.scatter3D(key[0], key[1], key[2], s = (KnOccurrences[key] / KnCount) * 10000, color =  'orange', alpha = 0.5, label = "KN population")
    
    for key in MdOccurrences:
        ax.scatter3D(key[0], key[1], key[2], s = (MdOccurrences[key] / MdCount) * 10000, color =  'blue', alpha = 0.5, label = "MD population")

    ax.set_xlabel('Predetection passband')
    ax.set_ylabel('Detection passband')
    ax.set_zlabel('Post Detection passband')
    plt.show()

dirToClassName = {
    'test_data/m-dwarf-flare-lightcurves/': 'M dwarf flares',
    'test_data/kasen-kilonova-lightcurves/DC_LSST_MODEL_KN17_WITH_HOST_EXT/': 'Kilo Nova',
}

# plotDetectionPassbands('test_data/m-dwarf-flare-lightcurves/','LSST_WFD_MODEL66_Mdwarf.LIST', True)
# plotDetectionPassbands('test_data/kasen-kilonova-lightcurves/DC_LSST_MODEL_KN17_WITH_HOST_EXT/', 'DC_LSST_MODEL_KN17_WITH_HOST_EXT.LIST', False)

# plotPreDetectionPassbands('test_data/m-dwarf-flare-lightcurves/','LSST_WFD_MODEL66_Mdwarf.LIST', True)
# plotPreDetectionPassbands('test_data/kasen-kilonova-lightcurves/DC_LSST_MODEL_KN17_WITH_HOST_EXT/', 'DC_LSST_MODEL_KN17_WITH_HOST_EXT.LIST', False)

# plotPostDetectionPassbands('test_data/m-dwarf-flare-lightcurves/','LSST_WFD_MODEL66_Mdwarf.LIST', True)
# plotPostDetectionPassbands('test_data/kasen-kilonova-lightcurves/DC_LSST_MODEL_KN17_WITH_HOST_EXT/', 'DC_LSST_MODEL_KN17_WITH_HOST_EXT.LIST', False)

# plotSuccessiveDetectionTimeDelta('test_data/m-dwarf-flare-lightcurves/','LSST_WFD_MODEL66_Mdwarf.LIST', True)
# plotSuccessiveDetectionTimeDelta('test_data/kasen-kilonova-lightcurves/DC_LSST_MODEL_KN17_WITH_HOST_EXT/', 'DC_LSST_MODEL_KN17_WITH_HOST_EXT.LIST', False)

# plotSignalToNoiseRatio('test_data/m-dwarf-flare-lightcurves/','LSST_WFD_MODEL66_Mdwarf.LIST', True)
# plotSignalToNoiseRatio('test_data/kasen-kilonova-lightcurves/DC_LSST_MODEL_KN17_WITH_HOST_EXT/', 'DC_LSST_MODEL_KN17_WITH_HOST_EXT.LIST', False)

# plotPreDetectionPassbandConfusionMatrix('test_data/m-dwarf-flare-lightcurves/','LSST_WFD_MODEL66_Mdwarf.LIST', True)
# plotPreDetectionPassbandConfusionMatrix('test_data/kasen-kilonova-lightcurves/DC_LSST_MODEL_KN17_WITH_HOST_EXT/', 'DC_LSST_MODEL_KN17_WITH_HOST_EXT.LIST', False)

# plotPostDetectionPassbandConfusionMatrix('test_data/m-dwarf-flare-lightcurves/','LSST_WFD_MODEL66_Mdwarf.LIST', True)
# plotPostDetectionPassbandConfusionMatrix('test_data/kasen-kilonova-lightcurves/DC_LSST_MODEL_KN17_WITH_HOST_EXT/', 'DC_LSST_MODEL_KN17_WITH_HOST_EXT.LIST', False)

plotAllObservationPassbandDistribution('test_data/m-dwarf-flare-lightcurves/','LSST_WFD_MODEL66_Mdwarf.LIST', True)
# plotAllObservationPassbandDistribution('test_data/kasen-kilonova-lightcurves/DC_LSST_MODEL_KN17_WITH_HOST_EXT/', 'DC_LSST_MODEL_KN17_WITH_HOST_EXT.LIST', False)

# plotPrePost3dPlot()