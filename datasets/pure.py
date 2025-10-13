import numpy as np
import os
import csv
import cv2
import json
from scipy.signal import medfilt
from . import Dataset, SignalData, CameraData
from ..signals import *

class PURE(Dataset):
    """
    PURE Dataset

    .. PURE dataset structure:
    .. -----------------
    ..     datasetDIR/
    ..     |
    ..     |-- 01-01/
    ..     |---- Image...1.png
    ..     |---- Image.....png
    ..     |---- Image...n.png
    ..     |-- 01-01.json
    ..     |...
    ..     |...
    ..     |-- nn-nn/
    ..     |---- Image...1.png
    ..     |---- Image.....png
    ..     |---- Image...n.png        
    ..     |-- nn-nn.json
    ..     |...
    """
    name = 'PURE'

    DATAPROPS = {
        'video':{
            'ext':'png',
            'rate':'timestamped',
            'compressed':False},
        'BVP':{
            'rate':'timestamped'},
        'ECG':None,
        'HR':{
            'rate':'timestamped'},
        'O2':{
            'rate':'timestamped'},
        'BR':'indirect',
    }
    

    DATA_SUFFIX = ".json"
    
    _SITUATION = ['Stationary','Talking','Slow Movement','Fast Movement','Small Rotation','Medium Rotation']
    
    def __init__(self, directory="D:/Datasets/PPG/PURE"):
        super().__init__(directory)
        
    def _init_filenames(self):
        root, dirs, files = next(os.walk(self.directory))
        for d in dirs:
            fpath = os.path.abspath(os.path.join(root, d))
            validity = self._get_gt_validity(fpath)
            if validity.upper() == "GOOD":
                self._video_paths.append(fpath)
            else:
                self._invalid_paths(fpath)
                
                
    #TODO other error checking of gt files
    def _get_gt_validity(self, videoPath):
        filename = videoPath + self.DATA_SUFFIX
        if os.path.isfile(filename):
            return "GOOD"
        else:
            return "MISSING"
        
    def get_instance_name(self, videoID=0, videoPath=None):
        videoPath = self._video_paths[videoID] if videoPath is None else videoPath
        _, name = os.path.split(videoPath)
        return name

    def get_friendly_name(self, videoID=0, videoPath=None):
        videoPath = self._video_paths[videoID] if videoPath is None else videoPath
        name = self.get_instance_name(videoPath=videoPath)
        name = name.split('-')
        subj = int(name[0])
        situ = self._SITUATION[int(name[1])]
        return "{} {}".format(situ, subj)

    def load_instance(self, videoID, include_video=True, clean_gt=True, gt_hr_params=None, **kwargs): 
        filename = self.get_video_path(videoID) + self.DATA_SUFFIX
        
        val_times = []
        bvp = []
        hr = []
        o2 = []
        video_times = []
        
        with open(filename) as json_file:
            json_data = json.load(json_file)
            for p in json_data['/FullPackage']:
                bvp.append(p['Value']['waveform'])
                hr.append(p['Value']['pulseRate'])
                o2.append(p['Value']['o2saturation'])
                val_times.append(p['Timestamp'])
            for p in json_data['/Image']:
                video_times.append(p['Timestamp'])
        
        val_times = np.array(val_times) / 1e9
        bvp = np.array(bvp)
        hr = np.array(hr)
        o2 = np.array(o2)
        if clean_gt:
            bvp = medfilt(bvp)
            hr = medfilt(hr)
            o2 = medfilt(o2)
            
        video_times = np.array(video_times) / 1e9
        
        start = min(val_times[0], video_times[0]) # Just taking for granted at the moment the fact that the differences between the starting and ending is less than a frame
        
        val_times -= start
        video_times -= start
        
        bvp = Signal(bvp, val_times)
        hr = Signal(hr, val_times)
        o2 = Signal(o2, val_times)
        
        if gt_hr_params is not None:
            hr = HR_from_BVP(bvp, **gt_hr_params)
        
        
        GT = SignalData(bvp=bvp,hr=hr,o2=o2)


        if include_video:
            return GT, CameraData.create(self._video_paths[videoID], video_times)
        else:
            return GT, video_times

class PURE_Compressed(PURE):
    def __init__(self, compression_str, compression_origin="D:/Datasets/PPG/CompressedCopies/PURE", directory="D:/Datasets/PPG/PURE"):
        super().__init__(directory)
        self.name = "PURE_Compressed@" + compression_str
        compression_origin = os.path.join(compression_origin.replace('/', os.sep),compression_str)
        directory = directory.replace('/', os.sep)
        self._compressed_video_paths = [x.replace(directory, compression_origin) + '.mp4' for x in self._video_paths]
        
        
        
    def load_instance(self, videoID, include_video=True, clean_gt=True, gt_hr_params=None, **kwargs): 
        filename = self.get_video_path(videoID) + self.DATA_SUFFIX
        
        val_times = []
        bvp = []
        hr = []
        o2 = []
        video_times = []
        
        with open(filename) as json_file:
            json_data = json.load(json_file)
            for p in json_data['/FullPackage']:
                bvp.append(p['Value']['waveform'])
                hr.append(p['Value']['pulseRate'])
                o2.append(p['Value']['o2saturation'])
                val_times.append(p['Timestamp'])
            for p in json_data['/Image']:
                video_times.append(p['Timestamp'])
        
        val_times = np.array(val_times) / 1e9
        bvp = np.array(bvp)
        hr = np.array(hr)
        o2 = np.array(o2)
        if clean_gt:
            bvp = medfilt(bvp)
            hr = medfilt(hr)
            o2 = medfilt(o2)
            
        video_times = np.array(video_times) / 1e9
        
        start = min(val_times[0], video_times[0]) # Just taking for granted at the moment the fact that the differences between the starting and ending is less than a frame
        
        val_times -= start
        video_times -= start
        
        bvp = Signal(bvp, val_times)
        hr = Signal(hr, val_times)
        o2 = Signal(o2, val_times)
        
        if gt_hr_params is not None:
            hr = HR_from_BVP(bvp, **gt_hr_params)
        
        
        GT = SignalData(bvp=bvp,hr=hr,o2=o2)


        if include_video:
            return GT, CameraData.create(self._compressed_video_paths[videoID], video_times)
        else:
            return GT, video_times