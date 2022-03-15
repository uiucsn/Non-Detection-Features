from datetime import time
from numpy.core.defchararray import less_equal
import matplotlib.pyplot as plt
import numpy as np
import NDFeatures
from collections import Counter
import pandas as pd
import sncosmo


def extractFeaturesFromFITS(path, listFile, isCompressed):
    
    dataFrames = []

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
                dataFrames.append(detectionData)
                fe.plotInstance()

    df = pd.concat(dataFrames)
    df['CLASS'] = [dirToClassName[path]] * len(df)
    df.to_csv(f'{dirToClassName[path]}_features.csv')
    print(df)

dirToClassName = {
    'test_data/m-dwarf-flare-lightcurves/': 'M dwarf flares',
    'test_data/kasen-kilonova-lightcurves/DC_LSST_MODEL_KN17_WITH_HOST_EXT/': 'Kilo Nova',
}

if __name__== '__main__':
    extractFeaturesFromFITS('test_data/m-dwarf-flare-lightcurves/','LSST_WFD_MODEL66_Mdwarf.LIST', True)
    extractFeaturesFromFITS('test_data/kasen-kilonova-lightcurves/DC_LSST_MODEL_KN17_WITH_HOST_EXT/', 'DC_LSST_MODEL_KN17_WITH_HOST_EXT.LIST', False)
