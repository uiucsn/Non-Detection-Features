import numpy as np
import matplotlib.pyplot as plt
from numpy.core.numeric import NaN

class NDFeatureExtractor:

    def __init__(self, dataFrame, survey):

        self.dataFrame = dataFrame
        self.survey = survey
        
    def extractDetectionData(self):
        
        detections = []
        idx = np.where(self.dataFrame['PHOTFLAG'] != 0)

        for i in idx:

            passband = self.dataFrame['BAND'][i]
            signal = self.dataFrame['FLUXCAL'][i]
            noise = self.dataFrame['FLUXCALERR'][i]
            
            detections.append((passband, signal, noise))
        
        self.features['Detection Data'] = detections
        return detections
    
    def extractTimeBetweenSucessiveDetections(self):
        
        timeDeltas = []
        idx = np.where(self.dataFrame['PHOTFLAG'] != 0)

        if len(idx) > 1:
            for i in range(1,idx):

                deltaTime = self.dataFrame['MJD'][i] - self.dataFrame['MJD'][i - 1]
                timeDeltas.append(deltaTime)
        
        self.features['Time Delta'] = timeDeltas
        return timeDeltas
    
    def extractSucceedingObservations(self, number = 1):

        succeedingObservations = []
        idx = np.where(self.dataFrame['PHOTFLAG'] != 0)

        for i in idx:
            
            temp = []

            # Ensuring that the end index is less than len - 1
            endIdx = i[0] + number
            while endIdx >= len(self.dataFrame) - 1:
                endIdx -= 1
            
            # Going through neigbouring observations
            for j in range(i[0] + 1, endIdx + 1):
                
                flag = self.dataFrame['PHOTFLAG'][j]
                passband = self.dataFrame['BAND'][j]
                signal = self.dataFrame['FLUXCAL'][j]
                noise = self.dataFrame['FLUXCALERR'][j]

                temp.append((flag, passband, signal, noise))

            if len(temp) > 0:
                succeedingObservations.append(temp)
        
        if len(succeedingObservations) == 0:
            succeedingObservations.append([(0, 'Invalid', 0, 0)])

        self.features['Succeeding Observations'] = succeedingObservations
        return succeedingObservations
    
    def extractPrecceedingObservations(self, number = 1):

        precceedingObservations = []
        idx = np.where(self.dataFrame['PHOTFLAG'] != 0)

        for i in idx:
            
            temp = []

            # Ensuring that the start index is greater than or equal to 0
            startIdx = i[0] - number
            while startIdx <= 0:
                startIdx += 1

            # Going through neigbouring observations
            for j in range(startIdx, i[0]):
                
                flag = self.dataFrame['PHOTFLAG'][j]
                passband = self.dataFrame['BAND'][j]
                signal = self.dataFrame['FLUXCAL'][j]
                noise = self.dataFrame['FLUXCALERR'][j]

                temp.append((flag, passband, signal, noise))
            
            if len(temp) > 0:
                precceedingObservations.append(temp)

        if len(precceedingObservations) == 0:
            precceedingObservations.append([(0, 'Invalid', 0, 0)])
        
        self.features['Precceeding Observations'] = precceedingObservations
        return precceedingObservations
    
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