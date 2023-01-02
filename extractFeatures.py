import pandas as pd
import os
from multiprocessing import Pool

mappings_bulla = pd.read_csv('test_data/MDF_VS_KN-KilonovaSims/Bulla/SNID_TO_SKYMAP.csv')
mappings_kasen = pd.read_csv('test_data/MDF_VS_KN-KilonovaSims/Kasen/SNID_TO_SKYMAP.csv')

def bulla_extract(i):

    print(f'Starting {i} out of {len(mappings_bulla)}')
    SNID = mappings_bulla['SNID'][i]
    output = f'Bulla_features/{SNID}.csv'
    skymap_path = f'test_data/MDF_VS_KN-KilonovaSims/Bulla/Bulla-Skymaps-Singleorder/{SNID}.singleorder.fits'
    
    os.system(f'python localize_flares.py {SNID} {skymap_path} {output} Bulla')
    print(f'Finished {i} out of {len(mappings_bulla)}')

def kasen_extract(i):

    print(f'Starting {i} out of {len(mappings_kasen)}')
    SNID = mappings_kasen['SNID'][i]
    output = f'Kasen_features/{SNID}.csv'
    skymap_path = f'test_data/MDF_VS_KN-KilonovaSims/Kasen/Kasen-Skymaps-Singleorder/{SNID}.singleorder.fits'
    
    os.system(f'python localize_flares.py {SNID} {skymap_path} {output} Kasen')
    print(f'Finished {i} out of {len(mappings_kasen)}')

if __name__ == '__main__':

    # print('Extracting features for Bulla sims...')
    # pool1 = Pool(os.cpu_count())
    # result = pool1.map(bulla_extract, range(len(mappings_bulla)))

    print('Extracting features for Kasen sims...')
    pool2 = Pool(os.cpu_count())
    result = pool2.map(kasen_extract, range(len(mappings_kasen)))

    print('Done!')
