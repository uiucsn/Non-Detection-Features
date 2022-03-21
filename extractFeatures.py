from datetime import time
from numpy.core.defchararray import less_equal
import matplotlib.pyplot as plt
import numpy as np
import NDFeatures
from collections import Counter
import pandas as pd
import sncosmo


def extractFeaturesFromFITS(path, listFile, isCompressed):
    
    # Storing the features from all the FITS files
    features = []

    with open(path + listFile) as file:
        for line in file:

            headFile = path + line.rstrip()
            photFile = headFile[:len(headFile) - 9] + 'PHOT.FITS'

            if isCompressed:
                headFile += '.gz'
                photFile += '.gz'
            
            # Collection of fits tables
            sims = sncosmo.read_snana_fits(headFile, photFile)

            for table in sims:

                fe = NDFeatures.NDFeatureExtractor(table, 'LSST')

                detectionData = fe.extractDetectionData()
                features.append(detectionData)
                fe.plotInstance()

    # Creating one df from all the df's and saving it
    df = pd.concat(features)
    df['CLASS'] = [dirToClassName[path]] * len(df)
    df.to_csv(f'{dirToClassName[path]}_features.csv')

dirToClassName = {
    'test_data/m-dwarf-flare-lightcurves/': 'M dwarf flares',
    'test_data/kasen-kilonova-lightcurves/DC_LSST_MODEL_KN17_WITH_HOST_EXT/': 'Kilo Nova',
}

if __name__== '__main__':
    extractFeaturesFromFITS('test_data/m-dwarf-flare-lightcurves/','LSST_WFD_MODEL66_Mdwarf.LIST', True)
    extractFeaturesFromFITS('test_data/kasen-kilonova-lightcurves/DC_LSST_MODEL_KN17_WITH_HOST_EXT/', 'DC_LSST_MODEL_KN17_WITH_HOST_EXT.LIST', False)
