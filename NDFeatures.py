from turtle import pos
import numpy as np
from numpy.core.defchararray import encode
import pandas as pd
import matplotlib.pyplot as plt
from numpy.core.numeric import NaN
import lightkurve as lk

class NDFeatureExtractor:

    # Data for internal use.
    passbandColors = {
        'LSST' : {'u ': 'tab:blue', 'g ': 'tab:orange', 'r ': 'tab:green', 'i ': 'tab:red', 'z ': 'tab:purple', 'Y ': 'tab:pink'},
    }

    def __init__(self, dataFrame, survey):

        self.dataFrame = dataFrame.to_pandas()
        self.dataFrame['BAND'] = np.array(self.dataFrame['BAND'], dtype=np.str)
        self.survey = survey
            
    def extractDetectionData(self, count = 1):
        """
        Returns a pandas dataframe with the detection, pre-dection and post-detection 
        passbands for all every detection in the FITS file along with the delta time 
        to the next detection (if any) and previous detection (if any). Maintains all
        the columns from the original FITS file but removes rows for non detections.

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

        # Extracting features
        pre_det_pb, post_det_pb = self.getPrePostDetPB(idx)
        timeToPrev, timeToNext = self.getTimeBetweenDetections(idx)

        # Adding PB data to the DF
        detectionDataFrame['PRE-BAND'] = pre_det_pb
        detectionDataFrame['POST-BAND'] = post_det_pb

        # Adding Delta T data to the DF
        detectionDataFrame['TIME-TO-PREV'] = timeToPrev
        detectionDataFrame['TIME-TO-NEXT'] = timeToNext
        
        # Returning sliced dataframe containing the correct number of detections.
        return detectionDataFrame[:count]

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

