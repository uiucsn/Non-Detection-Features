import numpy as np
from numpy.core.defchararray import encode
import pandas as pd
import matplotlib.pyplot as plt
from numpy.core.numeric import NaN
import lightkurve as lk

class NDFeatureExtractor:

    def __init__(self, dataFrame, survey):

        self.dataFrame = dataFrame.to_pandas()
        self.dataFrame['BAND'] = np.array(self.dataFrame['BAND'], dtype=np.str)
        self.survey = survey
        
    def extractDetectionData(self):

        detectionDataFrame = pd.DataFrame(columns = ['BAND', 'FLUXCAL', 'FLUXCALERR'])

        # Indexes for the detections
        boolArray = self.dataFrame['PHOTFLAG'] != 0
        idx = np.where(boolArray)[0]

        detectionDataFrame  = self.dataFrame.loc[idx, ]
        
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

            startIdx = i - number

            # Ensuring that the start index is greater than or equal to 0
            while startIdx < 0:
                startIdx += 1

            # A dataframe to store the suceeding observations of a given.
            preceedingObservationsDataFrame  = self.dataFrame.loc[range(startIdx, i),]
            arrayOfDataFrames.append(preceedingObservationsDataFrame)
        
        self.features['Precceeding Observations'] = arrayOfDataFrames
        return arrayOfDataFrames
    
    def extractSucceedingObservations(self, number = 1):
        
        # An array to hold all of the suceeding observations
        arrayOfDataFrames = []

        # Indexes for the detection
        boolArray = self.dataFrame['PHOTFLAG'] != 0
        idx = np.where(boolArray)[0]

        for i in idx:

            endIdx = i + number

            # Ensuring that the end index is less than len 
            while endIdx >= len(self.dataFrame):
                endIdx -= 1

            # A dataframe to store the suceeding observations of a given. 
            succeedingObservationsDataFrame  = self.dataFrame.loc[range(i + 1, endIdx + 1),]
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

        self.features['Signal to noise ratio'] = signalToNoiseRatioDF
        return signalToNoiseRatioDF
    
    def buildPseudoLightCurves(self, SNThreshold = 3):

        dict = self.passbandColors[self.survey]

        for band in dict.keys():

            df = self.dataFrame[self.dataFrame['FLUXCAL'] / self.dataFrame['FLUXCALERR'] >= SNThreshold] 
            df = df[df['BAND'].str.decode('utf-8') == band]

            signal = df['FLUXCAL']
            noise = df['FLUXCALERR']
            time = df['MJD']

            lc = lk.LightCurve(time = time, flux = signal, flux_err = noise)

            self.pseudoPassBandLightCurves[band] = lc


        return self.pseudoPassBandLightCurves

    def getFeatures(self):
        return self.features

    def plotInstance(self):

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
    
    def plotPseudoLightCurves(self, SNThreshold = 3):

        self.buildPseudoLightCurves(SNThreshold=SNThreshold)

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot()

        for band in self.passbandColors[self.survey].keys():
            lc = self.pseudoPassBandLightCurves[band]
            if len(lc) > 0:
                ax.errorbar(lc.time.to_value('mjd'), lc.flux.to_value(), yerr = self.pseudoPassBandLightCurves[band].flux_err.to_value(),  label = band, marker='o')
        
        plt.legend()
        plt.xlabel('Time in MJD')
        plt.ylabel('Flux')
        plt.show()

    features = {}

    # Storage for the pseudo light curves
    pseudoPassBandLightCurves = {}

    # Data for internal use.
    passbandColors = {
        'LSST' : {'u ': 'tab:blue', 'g ': 'tab:orange', 'r ': 'tab:green', 'i ': 'tab:red', 'z ': 'tab:purple', 'Y ': 'tab:pink'},
    }