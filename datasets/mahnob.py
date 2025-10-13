import xml.etree.ElementTree as ET
import numpy as np
import os
import warnings
from scipy.signal import medfilt
import re
import pyedflib
import cv2

from . import Dataset, SignalData, CameraData
from ..signals import *

# TODO seperate categories

class MAHNOB(Dataset):
    """
    Mahnob Dataset
    .. Mahnob dataset structure:
    .. -----------------
    ..     datasetDIR/
    ..     |
    ..     ||-- vidDIR1/
    ..     |   |-- videoFile.avi
    ..     |   |-- physioFile.bdf
    ..     |...
    ..     |...
    """
    name = 'MAHNOB'

    DATAPROPS = {
        'video':{
            'ext':'avi',
            'rate':'variable',
            'compressed':True},
        'BVP':None,
        'ECG':{
            'rate':256},
        'HR':None,
        'O2':'indirect',
        'BR':'indirect',
    }
    

    DATA_SUFFIX = ".bdf"
    
    def __init__(self, directory="D:/Datasets/PPG/MAHNOB-HCI"):
        super().__init__(directory)
        
    def _subject_trial(self, videoID=0, videoPath=None):
        videoPath = self._video_paths[videoID] if videoPath is None else videoPath
        fname = os.path.basename(videoPath)
        ints = [int(i) for i in re.findall(r'\d+', fname)]
        return ints[0], ints[-1]//2
        
    def get_instance_name(self, videoID=0, videoPath=None):
        videoPath = self._video_paths[videoID] if videoPath is None else videoPath
        subject, trial = self._subject_trial(videoPath=videoPath)
        return "S{}T{}".format(subject,trial)

    def get_friendly_name(self, videoID=0, videoPath=None):
        videoPath = self._video_paths[videoID] if videoPath is None else videoPath
        subject, trial = self._subject_trial(videoPath=videoPath)
        return "Subject {} Trial {}".format(subject, trial)
        
    #TODO other error checking of gt files
    def _get_gt_validity(self, videoPath):
        start_path = os.path.dirname(videoPath)
        subject, trial = self._subject_trial(videoPath=videoPath)
        filename = "Part_{}_S_Trial{}_emotion.bdf".format(subject,trial)
        filename = os.path.join(start_path, filename)
        if os.path.isfile(filename):
            return "GOOD"
        else:
            return "MISSING"
        
    def load_instance(self, videoID, include_video=True, clean_gt=True, gt_hr_params=None, **kwargs): 
        start_path = os.path.dirname(self._video_paths[videoID])
        subject, trial = self._subject_trial(videoID)
        fname = "Part_{}_S_Trial{}_emotion.bdf".format(subject,trial)
        fname = os.path.join(start_path, fname)
        
        edfFile = pyedflib.EdfReader(fname)
        start = np.where(edfFile.readSignal(46)>0)[0][0]

        ecg = edfFile.readSignal(33)[start:]
        hz = edfFile.samplefrequency(33)
        
        ecg = Signal(ecg, hz)
        
        hr = HR_from_ECG(ecg)
        
        GT = SignalData(ecg=ecg, hr=hr)
        
        if include_video:
            return GT, CameraData.create(self._video_paths[videoID])
        else:
            v = cv2.VideoCapture(self._video_paths[videoID])
            vframes = int(v.get(cv2.CAP_PROP_FRAME_COUNT))
            vlength = vframes/v.get(cv2.CAP_PROP_FPS)
            v.release()
            video_times = np.linspace(0,vlength,vframes)
            return GT, video_times
        
        # tree = ET.parse(os.path.join(start_path, "session.xml"))
        # tree = tree.getroot()
        
        #TODO
    
    
