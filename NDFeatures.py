import numpy as np
from numpy.core.defchararray import encode
import pandas as pd
import matplotlib.pyplot as plt
from numpy.core.numeric import NaN

class NDFeatureExtractor:

    def __init__(self, dataFrame, survey):

        self.dataFrame = dataFrame.to_pandas()
        self.survey = survey
        
    def extractDetectionData(self):

        detectionDataFrame = pd.DataFrame(columns = ['BAND', 'FLUXCAL', 'FLUXCALERR'])

        # Indexes for the detections
        boolArray = self.dataFrame['PHOTFLAG'] != 0
        idx = np.where(boolArray)[0]

        passband = np.array(self.dataFrame['BAND'][idx], dtype=np.str)
        signal = np.array(self.dataFrame['FLUXCAL'][idx])
        noise = np.array(self.dataFrame['FLUXCALERR'][idx])

        detectionDataFrame  = detectionDataFrame.append({'BAND': passband, 'FLUXCAL': signal, 'FLUXCALERR': noise}, ignore_index=True)
        
        self.features['Detection Data'] = detectionDataFrame
        return detectionDataFrame
    
    def extractTimeBetweenSucessiveDetections(self):
        
        timeDeltaDataFrame = pd.DataFrame(columns = ['Time Delta'])

        # Indexes for the detections.
        boolArray = self.dataFrame['PHOTFLAG'] != 0
        idx = np.where(boolArray)[0]

        # Only calculates time delta if there are more than one detections.
        if len(idx) > 1:
            for i in idx[1:]:

                deltaTime = self.dataFrame['MJD'][i] - self.dataFrame['MJD'][i - 1]
                timeDeltaDataFrame = timeDeltaDataFrame.append({'Time Delta': deltaTime}, ignore_index=True)
        
        self.features['Time Delta'] = timeDeltaDataFrame
        return timeDeltaDataFrame

    def extractPrecceedingObservations(self, number = 1):
        
        # An array to hold all of the precceeding observations
        arrayOfDataFrames = []

        # Indexes for the detection
        boolArray = self.dataFrame['PHOTFLAG'] != 0
        idx = np.where(boolArray)[0]

        for i in idx:

            # Ensuring that the start index is greater than or equal to 0
            startIdx = i - number
            while startIdx <= 0:
                startIdx += 1

            flag = np.array(self.dataFrame['PHOTFLAG'][range(startIdx, i)])
            passband = np.array(self.dataFrame['BAND'][range(startIdx, i)], dtype=np.str)
            signal = np.array(self.dataFrame['FLUXCAL'][range(startIdx, i)])
            noise = np.array(self.dataFrame['FLUXCALERR'][range(startIdx, i)])

            # A dataframe to store the suceeding observations of a given. 
            precceedingObservationsDataFrame = pd.DataFrame(columns = ['PHOTFLAG', 'BAND', 'FLUXCAL', 'FLUXCALERR'])
            precceedingObservationsDataFrame = precceedingObservationsDataFrame.append({'PHOTFLAG': flag, 'BAND': passband, 'FLUXCAL': signal, 'FLUXCALERR': noise}, ignore_index=True)

            arrayOfDataFrames.append(precceedingObservationsDataFrame)
        
        self.features['Precceeding Observations'] = arrayOfDataFrames
        return arrayOfDataFrames
    
    def extractSucceedingObservations(self, number = 1):
        
        # An array to hold all of the suceeding observations
        arrayOfDataFrames = []

        # Indexes for the detection
        boolArray = self.dataFrame['PHOTFLAG'] != 0
        idx = np.where(boolArray)[0]

        for i in idx:

            # Ensuring that the end index is less than len - 1
            endIdx = i + number
            while endIdx >= len(self.dataFrame) - 1:
                endIdx -= 1
            
            flag = np.array(self.dataFrame['PHOTFLAG'][range(i + 1, endIdx + 1)])
            passband = np.array(self.dataFrame['BAND'][range(i + 1, endIdx + 1)], dtype=np.str)
            signal = np.array(self.dataFrame['FLUXCAL'][range(i + 1, endIdx + 1)])
            noise = np.array(self.dataFrame['FLUXCALERR'][range(i + 1, endIdx + 1)])

            # A dataframe to store the suceeding observations of a given. 
            succeedingObservationsDataFrame = pd.DataFrame(columns = ['PHOTFLAG', 'BAND', 'FLUXCAL', 'FLUXCALERR'])
            succeedingObservationsDataFrame = succeedingObservationsDataFrame.append({'PHOTFLAG': flag, 'BAND': passband, 'FLUXCAL': signal, 'FLUXCALERR': noise}, ignore_index=True)

            arrayOfDataFrames.append(succeedingObservationsDataFrame)
        
        self.features['Succeeding Observations'] = arrayOfDataFrames
        return arrayOfDataFrames

    def extractSignalToNoiseRatio(self, number = 1):
        
        # An array to hold all of the n/s ratios
        signalToNoiseRatio = self.dataFrame['FLUXCAL'] / self.dataFrame['FLUXCALERR']
        
        passbands = np.array(self.dataFrame['BAND'], dtype=np.str)

        # A dataframe to store alll the signal to noise ratios along with the passbands
        signalToNoiseRatioDF = pd.DataFrame(columns = ['Signal to noise'])
        signalToNoiseRatioDF = signalToNoiseRatioDF.append({'Signal to noise': signalToNoiseRatio, 'BAND': passbands}, ignore_index=True)

        self.features['Succeeding Observations'] = signalToNoiseRatioDF
        return signalToNoiseRatioDF
    

    
    def getFeatures(self):
        return self.features

    def plotInstance(self):

        colors = self.passbandColors[self.survey]
        colorArr = []
        marker = []

        # Creating an array of colors based on passband.
        for passband in self.dataFrame['BAND']:
            colorArr.append(colors[passband])

        # Creating an array of markers based on whether the observation is a detection or not.
        for det in self.dataFrame['PHOTFLAG']:
            marker.append('o') if det == 0 else marker.append('^')

        # Creating the scatter plot.
        for i in range(len(self.dataFrame)):
            plt.errorbar(self.dataFrame['MJD'][i], self.dataFrame['FLUXCAL'][i], yerr = self.dataFrame['FLUXCALERR'][i], fmt = marker[i], c = colorArr[i], label = self.dataFrame['BAND'][i])

        plt.show()


    features = {}

    # Data for internal use.
    passbandColors = {
        'LSST' : {'u ': 'tab:blue', 'g ': 'tab:orange', 'r ': 'tab:green', 'i ': 'tab:red', 'z ': 'tab:purple', 'Y ': 'tab:pink'},
    }