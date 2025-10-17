import numpy as np
import os
import csv
import cv2
from scipy.signal import medfilt
from . import Dataset, SignalData, CameraData
from ..signals import *


### TODO

class PFF(Dataset):
    """
    PFF Dataset
    .. PFF dataset structure:
    .. permutations for x in {DM,DS,NM,NS,NSE} and y in {d, f, s}
    .. -----------------
    ..     datasetDIR/
    ..     |   |-- SubjDIR01/
    ..     |       |-- 01_x_y.avi
    ..     |       |-- 01_x_y.txt
    ..     |...
    ..     |   |-- SubjDIRM/
    ..     |       |-- M_x_y.avi
    ..     |       |-- M_x_y.txt
    """
    name = 'PFF'

    DATAPROPS = {
        'video':{
            'ext':'avi',
            'rate':'variable'},
        'BVP':None,
        'ECG':None,
        'HR':{
            'rate':1},
        'O2':'indirect',
        'BR':'indirect',
    }
    
    
    numLevels = 1             # depth of the filesystem collecting video and BVP files

    DATA_SUFFIX = "txt"

    _SITUATION_A = {'DM':"Dim Moving",'DS':'Dim Static', 'NM':'Moving', 'NS':'Static','NSE':'Exercise'}
    _SITUATION_B = {'d':'Upward Facing','f':'Forward Facing','s':'Side Facing'}
    
    def __init__(self, directory="D:/Datasets/PPG/PFF"):
        super().__init__(directory)

    def get_friendly_name(self, videoID):
        name = os.path.split(self._video_paths[videoID])[-1]
        spl = name[:-4].split('_')
        n = int(spl[0])
        a = self._SITUATION_A[spl[1]]
        b = self._SITUATION_B[spl[2]]
        return "{} ({}) {}".format(a,b,n)
        

    def load_instance(self, videoID, clean=True, include_video=True):
        pass
        # start_path = self._video_paths[videoID]
        # data_path = start_path[:-3] + self.DATA_SUFFIX
        

        # v = cv2.VideoCapture(self._video_paths[videoID])
        # vframes = int(v.get(cv2.CAP_PROP_FRAME_COUNT))
        # vlength = vframes/self.DATAPROPS.video.rate
        # v.release()

        # video_times = np.linspace(0,vlength,vframes)

        # val_times = []
        # hr = []
        
        # with open(data_path, 'r') as csvfile:
            # xmp = csv.reader(csvfile)
            # for row in xmp:
                # val_times.append(float(row[0])/1000.)
                # hr.append(float(row[1]))
                # o2.append(float(row[2]))
                # bvp.append(float(row[3]))

        # val_times = np.array(val_times)
        # bvp = np.array(bvp)
        # hr = np.array(hr)
        # o2 = np.array(o2)

        # if clean:
            # bvp = medfilt(bvp)
            # hr = medfilt(hr)
            # o2 = medfilt(o2)

        # GT = {
            
            # 'HR': Signal(hr, val_times, 'truth'),
            # 'O2': Signal(o2, val_times, 'truth')
            # }

        

        # return GT, video_times
