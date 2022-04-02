from datetime import time
from numpy.core.defchararray import less_equal
import matplotlib.pyplot as plt
import numpy as np
import NDFeatures
from collections import Counter
import pandas as pd
import sncosmo
from astropy.table import Table


def extractFeaturesFromFITS(path, listFile, skyMapPath, isCompressed):
    
    # Opening the GW skyMap
    skyMap = Table.read(skyMapPath)

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

                fe = NDFeatures.NDFeatureExtractor(table, skyMap, 0, 'LSST')

                detectionData = fe.extractDetectionData()
                features.append(detectionData)

    # Creating one df from all the df's and saving it
    df = pd.concat(features)
    df['CLASS'] = [dirToClassName[path]] * len(df)
    df.to_csv(f'{dirToClassName[path]}_features.csv')
    print(df)

dirToClassName = {
    'test_data/m-dwarf-flare-lightcurves/': 'M dwarf flares',
    'test_data/kasen-kilonova-lightcurves/DC_LSST_MODEL_KN17_WITH_HOST_EXT/': 'Kilo Nova',
}

if __name__== '__main__':
    #extractFeaturesFromFITS('test_data/m-dwarf-flare-lightcurves/','LSST_WFD_MODEL66_Mdwarf.LIST', 'bayestar.singleorder.FITS', True)
    extractFeaturesFromFITS('test_data/kasen-kilonova-lightcurves/DC_LSST_MODEL_KN17_WITH_HOST_EXT/', 'DC_LSST_MODEL_KN17_WITH_HOST_EXT.LIST', 'bayestar.singleorder.FITS', False)
