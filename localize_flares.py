from datetime import time
from numpy.core.defchararray import less_equal
import matplotlib.pyplot as plt
import numpy as np
import NDFeatures
from collections import Counter
import pandas as pd
import sncosmo
import healpy as hp
from astropy.table import Table
from astropy.coordinates import SkyCoord, ICRS
import astropy_healpix
from astropy import units as u
import argparse
import sys

KN_PHOT_FILES = {
    'Bulla': 'test_data/MDF_VS_KN-KilonovaSims/Bulla/BULLA_PHOT.FITS',
    'Kasen': 'test_data/MDF_VS_KN-KilonovaSims/Kasen/KASEN_PHOT.FITS',
}

KN_HEAD_FILES = {
    'Bulla': 'test_data/MDF_VS_KN-KilonovaSims/Bulla/BULLA_HEAD.FITS',
    'Kasen': 'test_data/MDF_VS_KN-KilonovaSims/Kasen/KASEN_HEAD.FITS',
}

map = astropy_healpix.HEALPix(32, frame=ICRS, order="nested")

def get_confidence_interval_mask(skymap, confidence_interval):
    """
    Create a mask for the region of the skymap that lies withing the 
    the mentioned confidence interval. 

    Args:
        skymap (fits Table): The sky map for the KN. Order must be nested. Any N side is allowed.
        confidence_interval (float): Confidence interval for which the map needs to be built.

    Returns:
        mask: If the healpix pixel lies inside the CI region, it is marked with a 1, else it's a 0.
        pixel indices: Healpix indices of the pixels lying in the Confidence interval region
    """

    prob = skymap["PROB"]
    sorted_prob_index = np.argsort(prob)

    # Finding the cumulative probability distribution for sorted prob values
    cum_sorted_prob = np.cumsum(prob[sorted_prob_index])

    # Searching for the min probability pixel such that cumulatibe proba is still CI
    threshold_index = np.searchsorted(cum_sorted_prob, 1 - confidence_interval)

    # Setting all pixels to 0
    mask = np.zeros(len(skymap))
    
    # If the pixels lie in high probability region, we set them to 1
    mask[sorted_prob_index[threshold_index:]] = 1

    return mask, set(sorted_prob_index[threshold_index:])

def getKNFeatures(skyMap, simType, SNID):

    # Find the KN fits file for the sky map (Same SNID)
    kn_fits = sncosmo.read_snana_fits(KN_HEAD_FILES[simType], KN_PHOT_FILES[simType], snids=[SNID])
    features = []

    for table in kn_fits:
        
        ra = table.meta['RA']
        dec = table.meta['DEC']

        fe = NDFeatures.NDFeatureExtractor(table, skyMap, 0, 'LSST')
        detectionData = fe.extractDetectionData()

        detectionData['RA'] = ra
        detectionData['DEC'] = dec
        features.append(detectionData)

    df = pd.concat(features)
    df['SNID'] = [table.meta['SNID']] * len(df)
    df['CLASS'] = [f'KN {simType}'] * len(df)
    return df

def getMDFFeatures(skyMap, high_ci_pixels):

    # Storing the features from all the FITS files
    features = []
    MDF_dir = 'test_data/m-dwarf-flare-lightcurves/'
    list_path = MDF_dir + 'LSST_WFD_MODEL66_Mdwarf.LIST'

    with open(list_path) as file:
        for line in file:
            
            headFile = MDF_dir + line.strip() + '.gz'
            photFile = headFile.replace('HEAD', 'PHOT')
            
            # Collection of fits tables
            sims = sncosmo.read_snana_fits(headFile, photFile)

            for table in sims:

                ra = table.meta['RA']
                dec = table.meta['DEC']

                c = SkyCoord(ra=ra, dec=dec, frame=ICRS, unit='deg')
                hp_index = map.skycoord_to_healpix(c, return_offsets=False)

                # Checking if flare is in high ci region
                if hp_index in high_ci_pixels:

                    fe = NDFeatures.NDFeatureExtractor(table, skyMap, 0, 'LSST')
                    detectionData = fe.extractDetectionData()
                    detectionData['RA'] = ra
                    detectionData['DEC'] = dec
                    detectionData['SNID'] = table.meta['SNID']
                    features.append(detectionData)

    # Creating one df from all the df's and saving it
    df = pd.concat(features)
    df['CLASS'] = ['MDF'] * len(df)
    return df

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

def extractFeaturesForSkyMap(SNID, skyMapPath, simType, output_path, ci = 0.9, fig=False):

    # Opening the GW skyMap
    skyMap = Table.read(skyMapPath)

    # Create high confidence interval mask
    mask, high_ci_pixels = get_confidence_interval_mask(skyMap, ci)

    # Find all MDF features for the sky map (high ci region)
    mdf_df = getMDFFeatures(skyMap, high_ci_pixels)

    # Find the KN features for the sky map (Same SNID)
    kn_df = getKNFeatures(skyMap, simType, SNID)

    # Combined features dataframe
    df = pd.concat([kn_df, mdf_df])
    df.to_csv(output_path)

    if fig == True:

        c = SkyCoord(ra=df['RA'], dec=df['DEC'], frame=ICRS, unit='deg')
        plotGenricSkyMap(c)

        # Make optional plot
        hp.mollview(np.log10(skyMap['PROB']), nest=True, min = -3, max = 0, title="Log Prob for GW event")
        hp.graticule(coord="E")

        hp.mollview(mask , nest=True, cmap='Greys',title="{}% Confidence interval healpix mask".format(ci * 100))
        hp.graticule(coord="E")

        plt.show()

if __name__=="__main__":
    
    def parse_args():
    
        # Getting Arguments
        argparser = argparse.ArgumentParser(
            description='Write files from db that lie within the CI of the skymap to a LCLIB file')

        argparser.add_argument('SNID', type=str, 
                            help='[REQUIRED] SNID of the simulation')
        argparser.add_argument('skymap_path', type=str, 
                            help='[REQUIRED] Path to the GW skymap FITS file (single order, nside = 32, nested)')
        argparser.add_argument('output', type=str,
                            help='[REQUIRED] Path to the output features file')
        argparser.add_argument('sim_type', type=str,
                            help='[REQUIRED] KN simulation type used for labelling. Must be one of [Bulla, Kasen]')
        argparser.add_argument('--ci', type=float, required=False, default=0.9,
                            help='Sky map confidence interval for localization. CI should be between 0 and 1 inclusive (Default = 0.9')
        argparser.add_argument('--plot', required=False, default=False, action='store_true',
                            help='Make a plot of the localization process')
        args = argparser.parse_args()

        if not args.sim_type in ['Bulla', 'Kasen']:
            print('KN simulation for skymap must be Kasen or Bulla. Aborting process.')
            sys.exit(1)

        if args.ci < 0 or args.ci > 1:
            print('CI should be between 0 and 1 inclusive. Aborting process.')
            sys.exit(1)

        return args

    args = parse_args()
    extractFeaturesForSkyMap(bytes(args.SNID, encoding='ascii'), args.skymap_path, args.sim_type, args.output, ci=args.ci, fig=args.plot)
    
    #extractFeaturesForSkyMap(b'1001622', 'test_data/MDF_VS_KN-KilonovaSims/Bulla/Bulla-Skymaps-Singleorder/1001622.singleorder.fits', 'Bulla', '1001622.csv')