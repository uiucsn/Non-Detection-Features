from turtle import pos
import numpy as np
import healpy as hp
from numpy.core.defchararray import encode
import pandas as pd
import matplotlib.pyplot as plt
from numpy.core.numeric import NaN
import lightkurve as lk
from astropy.coordinates import SkyCoord
from astropy.coordinates import ICRS
from astropy import units as u
import astropy_healpix


class NDFeatureExtractor:

    # 1. Data for internal use:

    ############################

    # NSIDE of the MDF map
    NSIDE = 32

    # Healpix index of the pixel that contains our event, rendered at NSIDE
    HEALPIX_INDEX = -1

    # Colors for the plots
    passbandColors = {
        'LSST' : {'u ': 'tab:blue', 'g ': 'tab:orange', 'r ': 'tab:green', 'i ': 'tab:red', 'z ': 'tab:purple', 'Y ': 'tab:pink'},
    }

    ############################

    def __init__(self, dataFrame, skyMap, survey):
        
        # Storing the RA and DEC for the instance
        self.ra = dataFrame.meta['RA']
        self.dec = dataFrame.meta['DEC']

        # Storing the skyMap
        self.skyMap = skyMap

        # Stroing as a pandas DF
        self.dataFrame = dataFrame.to_pandas()
        self.dataFrame['BAND'] = np.array(self.dataFrame['BAND'], dtype=np.str)

        # Density distribution map for m dwarf flares
        self.md_density_map = hp.read_map('data/m-dwarf-flare-density-1M-nside32.fits')

        self.survey = survey


    def extractDetectionData(self, count = 1):
        """
        Returns a pandas dataframe with the detection, pre-dection and post-detection 
        passbands for the first "count" detections in the FITS file along with the 
        delta time to the next detection (if any) and previous detection (if any). 
        Maintains all the columns from the original FITS file but removes rows for non 
        detections.

        Args:
            count (int): [Deafault = 1] Number of detections for which features should 
            be extracted. If count is 2, the dataframe with features for the first two 
            detections will be returned.

        Returns:
            Pandas data frame: Consists of the orignal data, passband data and delta
            times for all the detections in the FITS file.
        """

        # Indexes for the detections
        boolArray = self.dataFrame['PHOTFLAG'] != 0
        idx = np.where(boolArray)[0]

        detectionDataFrame  = self.dataFrame.loc[idx, ]

        # Adding PB data to the DF
        pre_det_pb, post_det_pb = self.getPrePostDetPB(idx)
        detectionDataFrame['PRE-BAND'] = pre_det_pb
        detectionDataFrame['POST-BAND'] = post_det_pb

        # Adding Delta T data to the DF
        timeToPrev, timeToNext = self.getTimeBetweenDetections(idx)
        detectionDataFrame['TIME-TO-PREV'] = timeToPrev
        detectionDataFrame['TIME-TO-NEXT'] = timeToNext

        # Adding next obs phot flag to the DF
        detectionDataFrame['NEXT-PHOT-FLAG'] = self.getNextObsPhotFlag(idx)

        # Adding the number of detections in the LC
        detectionDataFrame['NUM_DETECTIONS'] = self.getNumOfDetectionsInLC(idx)

        # Adding the M dwarf flare density for the healpix pixel where the event occured
        detectionDataFrame['MDF_DENSITY']  = self.getMDFlareDensity(idx)
        
        # Adding the GW probability for the healpix pixel where the event occured
        detectionDataFrame['GW_PROB'] = self.getGravitationalWaveProbability(idx)
        
        # Returning sliced dataframe containing the correct number of detections.
        return detectionDataFrame[:count]
    

    def getMDFlareDensity(self, idx):
        """
        Returns a list of the same length as idx containing the density of M dwarf flares in 
        the healpix pixel which contains our event, rendered with an NSIDE = 32.        

        Args:
            idx (numpy array): Indices of the detections in the FITS file.

        Returns:
            list: List containing the m dwarf density for the Healpix pixel containing the event
            repeated len(idx) times.
        """

        # Converting the coordinates to the HELPIX pixel
        coordinates = SkyCoord(ra = self.ra * u.deg, dec = self.dec * u.deg, frame=ICRS)
        map = astropy_healpix.HEALPix(self.NSIDE, frame=ICRS, order="nested")
        healpix_index = map.skycoord_to_healpix(coordinates, return_offsets=False)
        
        # Finding the density of mdwarf in this pixel
        pixel_prob = self.md_density_map[healpix_index]

        pixel_prob_list = [pixel_prob] * len(idx)

        return pixel_prob_list


    def getNumOfDetectionsInLC(self, idx):
        """
        Returns a list with length equal to the number of detection containing the
        number of detections in the LC. Consequently, all the values in the list 
        will have the same value.

        Args:
            idx (numpy array): Indices of the detections in the FITS file.

        Returns:
            list: List containing the number of detections in the LC, repeated len(idx)
            times.
        """

        numDetections = len(idx)
        numberOfDetectionsColumns = [numDetections] * numDetections 
        return numberOfDetectionsColumns


    def getNextObsPhotFlag(self, idx):
        """
        Returns a list with length equal to idx. If the next observation for a detection
        from idx is also a detection, the value will be 1. Otherwise, the value will be 
        0.

        Args:
            idx (numpy array): Indices of the detections in the FITS file.

        Returns:
            list : List containing binary values relating to the next observations PHOT 
            FLAG.
        """

        nextObsPhotFlag = []

        for i in idx:

            next_index = i + 1

            # Adding post detection flag if it exists
            if next_index >= len(self.dataFrame):
                nextObsPhotFlag.append(0)
            else:
                flag = self.dataFrame['PHOTFLAG'][next_index]
                if flag != 0:
                    nextObsPhotFlag.append(1)
                else:
                    nextObsPhotFlag.append(0)

        return nextObsPhotFlag
            

    def getPrePostDetPB(self, idx):
        """
        Returns two lists containing the pre - detection and post - detection passbands 
        for every detection in the FITS file. 

        Args:
            idx (numpy array): Indices of the detections in the FITS file.

        Returns:
            List : pre_det_pb - list containing the pre - detection passbands for all 
            detections. The entry is an empty string if there is no non detection before 
            the detection.
            List : post_det_pb - list containing the post - detection passbands for all 
            detections. The entry is an empty string if there is no non detection after 
            the detection.
        """

        # Predetection observations.
        pre_det_pb = []

        # Postdetection observations.
        post_det_pb = []

        # Getting pre and post detection passbands for every detection
        for i in range(len(idx)):

            pre_index = idx[i] - 1
            post_index = idx[i] + 1

            # Adding pre detection passband if it exists
            if pre_index < 0:
                pre_det_pb.append(None)
            else:
                pre_det_pb.append(self.dataFrame['BAND'][pre_index])

            # Adding post detection passband if it exists
            if post_index >= len(self.dataFrame):
                post_det_pb.append(None)
            else:
                post_det_pb.append(self.dataFrame['BAND'][post_index])
        
        return pre_det_pb, post_det_pb
    

    def getTimeBetweenDetections(self, idx):
        """
        Returns two lists containing the time to previous detection and time to next
        detection for every detection in the FITS file.         

        Args:
            idx (numpy array): Indices of the detections in the FITS file.

        Returns:
            List: timeToPrev - Contains the time to previous detection. Value is
            None if there is no previous detection.
            List: timeToNext - Contains the time to next detection. Value is None
            if there is no previous detection.
        """

        timeToNext = []
        timeToPrev = []

        for i in range(len(idx)):
            
            # Time to next detection
            if i == len(idx) - 1:
                # If it is the last detection, add a special value for next.
                timeToNext.append(-1)
            else:
                delta_time = self.dataFrame['MJD'][i + 1] - self.dataFrame['MJD'][i]
                timeToNext.append(delta_time)

            # Time to previous detection
            if i == 0:
                # If it is first detection, add a special value for previous.
                timeToPrev.append(-1)
            else:
                delta_time = self.dataFrame['MJD'][i] - self.dataFrame['MJD'][i - 1]
                timeToPrev.append(delta_time)

        return timeToPrev, timeToNext

    def getGravitationalWaveProbability(self, idx):
        """
        Returns a list of the same length as idx containing the probability of GW event in the 
        the healpix pixel which contains our event, rendered with an NSIDE = 32.        

        Args:
            idx (numpy array): Indices of the detections in the FITS file.

        Returns:
            list: List containing the GW probability for the Healpix pixel containing the event
            repeated len(idx) times.
        """

        # Converting the coordinates to the HELPIX pixel
        coordinates = SkyCoord(ra = self.ra * u.deg, dec = self.dec * u.deg, frame=ICRS)
        map = astropy_healpix.HEALPix(self.NSIDE, frame=ICRS, order="nested")
        healpix_index = map.skycoord_to_healpix(coordinates, return_offsets=False)

        # Finding the GW probability associated with the pixel
        prob = self.skyMap["PROB"][healpix_index]

        # Creating a list with repeated value
        probability_list = [prob] * len(idx)

        return probability_list

    def plotInstance(self):
        """
        Plots the instance of the light curve from the FITS file. 
        """

        colors = self.passbandColors[self.survey]
        colorArr = []
        marker = []

        # Creating an array of colors based on passband.
        for passband in np.array(self.dataFrame['BAND'], dtype=np.str):
            colorArr.append(colors[passband])

        # Creating an array of markers based on whether the observation is a detection or not.
        for det in self.dataFrame['PHOTFLAG']:
            marker.append('o') if det == 0 else marker.append('^')

        # Creating the scatter plot.
        for i in range(len(self.dataFrame)):
            plt.errorbar(self.dataFrame['MJD'][i], self.dataFrame['FLUXCAL'][i], yerr = self.dataFrame['FLUXCALERR'][i], fmt = marker[i], c = colorArr[i], label = self.dataFrame['BAND'][i])
        
        plt.show()

