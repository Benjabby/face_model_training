import numpy as np
import os
import h5py
from scipy.signal import medfilt, find_peaks
from scipy.ndimage import uniform_filter
import cv2


from . import Dataset, SignalData, CameraData
from ..signals import *
from vsmodule.consts import Globals


######### TODO

class COHFACE(Dataset):
    """
    Cohface Dataset
    
    .. Cohface dataset structure:
    .. -----------------
    ..    datasetDIR/
    ..    |
    ..    |-- subjDIR_1/
    ..    |   |-- vidDIR1/
    ..    |       |-- videoFile1.avi
    ..    |       |-- ...
    ..    |       |-- videoFileN.avi
    ..    |...
    ..    |   |-- vidDIRM/
    ..    |       |-- videoFile1.avi
    ..    |       |-- ...
    ..    |       |-- videoFileM.avi
    ..    |...
    ..    |-- subjDIR_n/
    ..    |...
    """
    name = 'COHFACE'

    DATAPROPS = {
        'video':{
            'ext':'avi',
            'rate':20,
            'compressed':True},
        'BVP':{
            'rate':256},
        'ECG':None,
        'HR':'indirect',
        'O2':'indirect',
        'BR':'indirect',
    }

    DATA_SUFFIX = "data.hdf5"
    
    def __init__(self, directory="D:/Datasets/PPG/COHFACE"):
        super().__init__(directory)
        
    def get_instance_name(self, videoID=0, videoPath=None):
        videoPath = self._video_paths[videoID] if videoPath is None else videoPath
        x, name1 = os.path.split(os.path.dirname(videoPath))
        _, name2 = os.path.split(x)
        return name2+'-'+name1
        
    def get_friendly_name(self, videoID=0, videoPath=None):
        videoPath = self._video_paths[videoID] if videoPath is None else videoPath
        x, name1 = os.path.split(os.path.dirname(videoPath))
        _, name2 = os.path.split(x)
        
        name = "Subject " + name2 + " Video " + name1
        return name
        
    #TODO other error checking of gt files
    def _get_gt_validity(self, videoPath):
        start_path = os.path.dirname(videoPath)
        filename = os.path.join(start_path, self.DATA_SUFFIX)
        if os.path.isfile(filename):
            return "GOOD"
        else:
            return "MISSING"

    def load_instance(self, videoID, include_video=True, clean_gt=True, gt_hr_params=None, **kwargs): 
        start_path = os.path.dirname(self._video_paths[videoID])
        
        filename = os.path.join(start_path, self.DATA_SUFFIX)
        
        f = h5py.File(filename, 'r')
        bvp = np.array(f['pulse'])
        rbvp = np.array(f['respiration'])
        times = np.array(f['time'])
        
        if clean_gt:
            bvp = medfilt(bvp)
            rbvp = medfilt(rbvp)
        
        bvp = Signal(bvp, times)
        
        if gt_hr_params is None:
            hidx, _ = find_peaks(bvp.data, distance=self.DATAPROPS.BVP.rate/Globals.HR_MAX_HZ, prominence=0.5, height=40)
            hr = 60/np.diff(times[hidx])
            hr = uniform_filter(hr,3)
            hr_time = times[hidx][1:] - np.diff(times[hidx])/2 # Place the values in between the two peaks
            hr = Signal(hr, hr_time)
        else:
            hr = HR_from_BVP(bvp, **gt_hr_params)
            
        
        bidx, _ = find_peaks(rbvp, distance=self.DATAPROPS.BVP.rate/Globals.BR_MAX_HZ, prominence=0.5)
        br = 60/np.diff(times[bidx])
        br = uniform_filter(br,3)
        br_time = times[bidx][1:] - np.diff(times[bidx])/2 # Place the values in between the two peaks
        br = Signal(br, br_time)

        GT = SignalData(bvp=bvp,hr=hr,br=br)

        if include_video:
            return GT, CameraData.create(self._video_paths[videoID])
        else:
            v = cv2.VideoCapture(self._video_paths[videoID])
            vframes = int(v.get(cv2.CAP_PROP_FRAME_COUNT))
            vlength = vframes/self.DATAPROPS.video.rate
            v.release()

            video_times = np.linspace(0,vlength,vframes)
            return GT, video_times
    
    
