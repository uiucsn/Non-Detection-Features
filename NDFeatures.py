from _typeshed import Self
import pandas as pd
import numpy as np

class NonDetectionFeatures:

    unifiedLightCurve = pd.DataFrame(columns=['passband' , 'flux', 'fluxError', 'detection', 'time'])


    def __init__(self, lcDict):

        self.lcDict = lcDict
        
    def collapseLightCurves(self):

        # Going through each passband in the dictionary
        for passband in self.lcDict.keys():
            lc = self.lcDict[passband];
            # Adding every detection and non detection to a dataframe
            for i in len(lc):
                detection = (lc.flux[i] / lc.error[i]) > 5
                self.unifiedLightCurve.append([[passband, lc.flux[i], lc.error[i], detection, lc.time[i]]])

        # Sorting the dataframe by time
        self.unifiedLightCurve.sort_values('time', ignore_index=True)
    
