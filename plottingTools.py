from datetime import time
from numpy.core.defchararray import less_equal
import sncosmo
import matplotlib.pyplot as plt
import numpy as np
import NDFeatures
from collections import Counter
import pandas as pd


def plotFirstDetectionPassbands(path, listFile, isCompressed):
    
    detectionPassband = []

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
                detectionPassband.append(detectionData[0][0][0])
            

    detectionPassband = sorted(detectionPassband)
                
    letter_counts = Counter(detectionPassband)
    df = pd.DataFrame.from_dict(letter_counts, orient='index')
    df = df.reindex(['u ','g ','r ','i ','z ','Y '])
    df.plot(kind='bar')

    plt.title('Distribution of first detection passbands ' + dirToClassName[path])
    plt.show()

def plotPreDetectionPassbands(path, listFile, isCompressed):

    
    preDetectionPassband = []

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

                detectionData = fe.extractPrecceedingObservations()
                preDetectionPassband.append(detectionData[0][0][1]) 

    preDetectionPassband = np.array(preDetectionPassband)
    preDetectionPassband = preDetectionPassband[preDetectionPassband != 'Invalid']
    preDetectionPassband = sorted(preDetectionPassband)

    letter_counts = Counter(preDetectionPassband)
    df = pd.DataFrame.from_dict(letter_counts, orient='index')
    df = df.reindex(['u ','g ','r ','i ','z ','Y '])
    df.plot(kind='bar')

    plt.title('Distribution of pre detection passbands ' + dirToClassName[path])
    plt.show()

def plotPostDetectionPassbands(path, listFile, isCompressed):
    
    postDetectionPassbands = []

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

                detectionData = fe.extractSucceedingObservations()
                postDetectionPassbands.append(detectionData[0][0][1])   

    postDetectionPassbands = np.array(postDetectionPassbands)
    postDetectionPassbands = postDetectionPassbands[postDetectionPassbands != 'Invalid']
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
                if len(timeBetweenDetections) > 0:
                    timeDelta.append(timeBetweenDetections[0])

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