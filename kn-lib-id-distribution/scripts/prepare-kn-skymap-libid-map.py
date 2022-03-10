from functools import partial
import logging
import multiprocessing
from glob import glob
from os import path

from astropy.coordinates import SkyCoord
from ligo.skymap.io import read_sky_map
from ligo.skymap.postprocess import crossmatch
import numpy as np
import pandas as pd

log = logging.getLogger('skymap-lib-id-job')
log.setLevel(logging.INFO)
file_handler = logging.FileHandler('log.log')
file_handler.setLevel(logging.INFO)
log.addHandler(file_handler)


def get_crossmatch(skymap_filename, coords, lib_id_coordinate_map):
    skymap = read_sky_map(skymap_filename, moc=True)
    res = crossmatch(skymap, coords)
    log = logging.getLogger('skymap-lib-id-job')
    log.info(f"Done crossmatching for {skymap_filename}")
    return (
        lib_id_coordinate_map.loc[
            res.searched_prob < 0.9].lib_id.values,
        skymap_filename
    )


skymap_directory = """/projects/ncsa/caps/deep/uiucsn/retraining-rapid/kn-rapid/utils/skymaps-ztf-simlib-bulla-O4"""
skymap_filenames = glob(path.join(skymap_directory, "*.fits"))
log.info("Loaded skymaps")
lib_id_coordinate_map = pd.read_pickle("lib_id_kn_skymap_relation/ztf_dr3_lib_id_map.pickle")

coords = SkyCoord(lib_id_coordinate_map.ra,
                  lib_id_coordinate_map.dec,
                  unit='deg')
log.info("Loaded LIBID coordinates")
target = partial(get_crossmatch, coords=coords,
                 lib_id_coordinate_map=lib_id_coordinate_map)
res_objs = list()
results =  list()
with multiprocessing.Pool(40) as p:
    res_objs = [
        p.apply_async(target, (skymap_filename,))
        for skymap_filename in skymap_filenames
    ]

    for r in res_objs:
        _r, skymap_filename = r.get()
        results.append((skymap_filename, _r))
pd.DataFrame(
    data=results, columns=('skymap_filename', 'lib_ids'),
).to_pickle('result.pickle')

