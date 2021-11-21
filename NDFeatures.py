import numpy as np
import matplotlib.pyplot as plt

class NDFeatureExtractor:

    def __init__(self, dataFrame, survey):

        self.dataFrame = dataFrame
        self.survey = survey
        
    def plotInstance(self):

        colors = self.passbandColors[self.survey]
        colorArr = []
        marker = []

        # Creating an array of colors based on passband.
        for passband in self.dataFrame['BAND']:
            colorArr.append(colors[passband])

        # Creating an array of markers based on whether the observation is a detection or not.
        for det in self.dataFrame['PHOTFLAG']:
            marker.append(u'o') if det == 0 else marker.append(u'^')

        # Creating the scatter plot.
        for i in range(len(self.dataFrame)):
            plt.errorbar(self.dataFrame['MJD'][i], self.dataFrame['FLUXCAL'][i], yerr = self.dataFrame['FLUXCALERR'][i], fmt = marker[i], c = colorArr[i], label = self.dataFrame['BAND'][i])

        plt.show()


    # Data for internal use.
    passbandColors = {
        'LSST' : {'u ': 'tab:blue', 'g ': 'tab:orange', 'r ': 'tab:green', 'i ': 'tab:red', 'z ': 'tab:purple', 'Y ': 'tab:pink'},
    }