from datetime import time
from numpy.core.defchararray import less_equal
import sncosmo
import matplotlib.pyplot as plt
import numpy as np
import NDFeatures
from collections import Counter
import pandas as pd


def plotFirstDetectionPassbands(path, listFile, isCompressed):
    
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

                detectionPassband = np.concatenate([detectionPassband, detectionData['BAND'].to_numpy()[0]])
        
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
                    preDetectionPassband = np.concatenate([preDetectionPassband, df['BAND'].to_numpy()[0]])

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
                    postDetectionPassbands = np.concatenate([postDetectionPassbands, df['BAND'].to_numpy()[0]])   

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

dirToClassName = {
    'test_data/m-dwarf-flare-lightcurves/': 'M dwarf flares',
    'test_data/kasen-kilonova-lightcurves/DC_LSST_MODEL_KN17_WITH_HOST_EXT/': 'Kilo Nova',
}

# plotFirstDetectionPassbands('test_data/m-dwarf-flare-lightcurves/','LSST_WFD_MODEL66_Mdwarf.LIST', True)
# plotFirstDetectionPassbands('test_data/kasen-kilonova-lightcurves/DC_LSST_MODEL_KN17_WITH_HOST_EXT/', 'DC_LSST_MODEL_KN17_WITH_HOST_EXT.LIST', False)

# plotPreDetectionPassbands('test_data/m-dwarf-flare-lightcurves/','LSST_WFD_MODEL66_Mdwarf.LIST', True)
# plotPreDetectionPassbands('test_data/kasen-kilonova-lightcurves/DC_LSST_MODEL_KN17_WITH_HOST_EXT/', 'DC_LSST_MODEL_KN17_WITH_HOST_EXT.LIST', False)

# plotPostDetectionPassbands('test_data/m-dwarf-flare-lightcurves/','LSST_WFD_MODEL66_Mdwarf.LIST', True)
# plotPostDetectionPassbands('test_data/kasen-kilonova-lightcurves/DC_LSST_MODEL_KN17_WITH_HOST_EXT/', 'DC_LSST_MODEL_KN17_WITH_HOST_EXT.LIST', False)

# plotSuccessiveDetectionTimeDelta('test_data/m-dwarf-flare-lightcurves/','LSST_WFD_MODEL66_Mdwarf.LIST', True)
# plotSuccessiveDetectionTimeDelta('test_data/kasen-kilonova-lightcurves/DC_LSST_MODEL_KN17_WITH_HOST_EXT/', 'DC_LSST_MODEL_KN17_WITH_HOST_EXT.LIST', False)