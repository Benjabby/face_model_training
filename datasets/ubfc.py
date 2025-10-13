import numpy as np
import os
import csv
import cv2
from . import Dataset, SignalData, CameraData
from ..signals import Signal, HR_from_BVP, BR_from_BVP
from scipy.signal import medfilt

class UBFC1(Dataset):
    """
    UBFC1 Dataset
    .. UBFC dataset structure:
    .. -----------------
    ..     datasetDIR/
    ..     |   |-- SubjDIR1/
    ..     |       |-- vid.avi
    ..     |...
    ..     |   |-- SubjDIRM/
    ..     |       |-- vid.avi
    """
    name = 'UBFC1'

    DATAPROPS = {
        'video':{
            'ext':'avi',
            'rate':28.671786,
            'compressed':False},
        'BVP':{
            'rate':'timestamped'},
        'ECG':None,
        'HR':{
            'rate':'timestamped',
            'lag':4},
        'O2':{
            'rate':'timestamped',
            'lag':4},
        'BR':'indirect',
    }

    DATA_SUFFIX = "gtdump.xmp"
    
    def __init__(self, directory="D:/Datasets/PPG/UBFC1"):
        super().__init__(directory)
        
    def get_instance_name(self, videoID=0, videoPath=None):
        videoPath = self._video_paths[videoID] if videoPath is None else videoPath
        _, name = os.path.split(os.path.dirname(videoPath))
        return name
        

    def get_friendly_name(self, videoID=0, videoPath=None):
        videoPath = self._video_paths[videoID] if videoPath is None else videoPath
        name = self.get_instance_name(videoPath=videoPath)
        if name == 'after-exercise':
            return "Subject 10 (After Exercise)"
        else:
            return "Subject " + name.removesuffix('-gt')
            
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
        data_path = os.path.join(start_path, self.DATA_SUFFIX)

        val_times = []
        bvp = []
        hr = []
        o2 = []
        
        with open(data_path, 'r') as csvfile:
            xmp = csv.reader(csvfile)
            for row in xmp:
                val_times.append(float(row[0])/1000.)
                hr.append(float(row[1]))
                o2.append(float(row[2]))
                bvp.append(float(row[3]))

        val_times = np.array(val_times)
        bvp = np.array(bvp)
        if clean_gt: bvp = medfilt(bvp)
        bvp = Signal(bvp, val_times)
        
        if gt_hr_params is None:
            hr = np.array(hr)
            hr_times = val_times-self.DATAPROPS.HR.lag
            st = np.searchsorted(hr_times, 0, 'right')
            hr_times = hr_times[st:]
            hr = hr[st:]
            if clean_gt: hr = medfilt(hr)
            hr = Signal(hr, hr_times)
        else:
            hr = HR_from_BVP(bvp, **gt_hr_params)
        
        o2 = np.array(o2)
        o2_times = val_times-self.DATAPROPS.O2.lag
        st = np.searchsorted(o2_times, 0, 'right')
        o2_times = o2_times[st:]
        o2 = o2[st:]
        if clean_gt: o2 = medfilt(o2)
        o2 = Signal(o2, o2_times)

        if 'gt_br_params' in kwargs and kwargs['gt_br_params'] is not None:
            br = BR_from_BVP(bvp, **kwargs['gt_br_params'])
        else:
            br = BR_from_BVP(bvp)
        
        GT = SignalData(bvp=bvp, hr=hr, o2=o2, br=br)

        
        if include_video:
            return GT, CameraData.create(self._video_paths[videoID])
        else:
            v = cv2.VideoCapture(self._video_paths[videoID])
            vframes = int(v.get(cv2.CAP_PROP_FRAME_COUNT))
            vlength = vframes/self.DATAPROPS.video.rate
            v.release()
            video_times = np.linspace(0,vlength,vframes)
            return GT, video_times
            
    
class UBFC2(Dataset):
    """
    UBFC2 Dataset

    .. UBFC dataset structure:
    .. -----------------
    ..     datasetDIR/
    ..     |   |-- SubjDIR1/
    ..     |       |-- vid.avi
    ..     |...
    ..     |   |-- SubjDIRM/
    ..     |       |-- vid.avi
    """
    name = 'UBFC2'

    DATAPROPS = {
        'video':{
            'ext':'avi',
            'rate':'variable',
            'compressed':False},
        'BVP':{
            'rate':'timestamped'},
        'ECG':None,
        'HR':{
            'rate':'timestamped'},
        'O2':'indirect',
        'BR':'indirect',
    }
    
    
    numLevels = 2             # depth of the filesystem collecting video and BVP files

    DATA_SUFFIX = "ground_truth.txt"
    
    def __init__(self, directory="D:/Datasets/PPG/UBFC2"):
        super().__init__(directory)
        
    def get_instance_name(self, videoID=0, videoPath=None):
        videoPath = self._video_paths[videoID] if videoPath is None else videoPath
        _, name = os.path.split(os.path.dirname(videoPath))
        return name

    def get_friendly_name(self, videoID=0, videoPath=None):
        videoPath = self._video_paths[videoID] if videoPath is None else videoPath
        name = self.get_instance_name(videoPath=videoPath)
        return name.replace('subject', 'Subject ')
        
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
        data_path = os.path.join(start_path, self.DATA_SUFFIX)

        
        with open(data_path, 'r') as f:
           x = f.readlines()

        bvp = x[0].split(' ')
        bvp = list(filter(lambda a: a != '', bvp))
        bvp = np.array(bvp).astype(np.float64)
        if clean_gt: bvp = medfilt(bvp)
        
        val_times = x[2].split(' ')
        val_times = list(filter(lambda a: a != '', val_times))
        val_times = np.array(val_times).astype(np.float64)
        #val_times = val_times - val_times[0]
        
        bvp = Signal(bvp, val_times)

        if gt_hr_params is None:
            hr = x[1].split(' ')
            hr = list(filter(lambda a: a != '', hr))
            hr = np.array(hr).astype(np.float64)
            if clean_gt: hr = medfilt(hr)
            hr = Signal(hr, val_times)
        else:
            hr = HR_from_BVP(bvp, **gt_hr_params)
            

        GT = SignalData(bvp=bvp, hr=hr)

        
        if include_video:
            return GT, CameraData.create(self._video_paths[videoID])
        else:
            v = cv2.VideoCapture(self._video_paths[videoID])
            vframes = int(v.get(cv2.CAP_PROP_FRAME_COUNT))
            vlength = vframes/v.get(cv2.CAP_PROP_FPS)
            v.release()
            video_times = np.linspace(0,vlength,vframes)
            return GT, video_times


class UBFC1_Compressed(UBFC1):
    def __init__(self, compression_str, compression_origin="D:/Datasets/PPG/CompressedCopies/UBFC1", directory="D:/Datasets/PPG/UBFC1"):
        super().__init__(directory)
        self.name = "UBFC1_Compressed@" + compression_str
        compression_origin = os.path.join(compression_origin.replace('/', os.sep),compression_str)
        directory = directory.replace('/', os.sep)
        self._compressed_video_paths = [x.replace(directory, compression_origin)[:-3] + 'mp4' for x in self._video_paths]
    
    def get_video_path(videoID):
        return self._compressed_video_paths[videoID]
    
    def load_instance(self, videoID, include_video=True, clean_gt=True, gt_hr_params=None, **kwargs):
        start_path = os.path.dirname(self._video_paths[videoID])
        data_path = os.path.join(start_path, self.DATA_SUFFIX)

        val_times = []
        bvp = []
        hr = []
        o2 = []
        
        with open(data_path, 'r') as csvfile:
            xmp = csv.reader(csvfile)
            for row in xmp:
                val_times.append(float(row[0])/1000.)
                hr.append(float(row[1]))
                o2.append(float(row[2]))
                bvp.append(float(row[3]))

        val_times = np.array(val_times)
        bvp = np.array(bvp)
        if clean_gt: bvp = medfilt(bvp)
        bvp = Signal(bvp, val_times)
        
        if gt_hr_params is None:
            hr = np.array(hr)
            hr_times = val_times-self.DATAPROPS.HR.lag
            st = np.searchsorted(hr_times, 0, 'right')
            hr_times = hr_times[st:]
            hr = hr[st:]
            if clean_gt: hr = medfilt(hr)
            hr = Signal(hr, hr_times)
        else:
            hr = HR_from_BVP(bvp, **gt_hr_params)
        
        o2 = np.array(o2)
        o2_times = val_times-self.DATAPROPS.O2.lag
        st = np.searchsorted(o2_times, 0, 'right')
        o2_times = o2_times[st:]
        o2 = o2[st:]
        if clean_gt: o2 = medfilt(o2)
        o2 = Signal(o2, o2_times)

        if 'gt_br_params' in kwargs and kwargs['gt_br_params'] is not None:
            br = BR_from_BVP(bvp, **kwargs['gt_br_params'])
        else:
            br = BR_from_BVP(bvp)
        
        GT = SignalData(bvp=bvp, hr=hr, o2=o2, br=br)

        
        if include_video:
            return GT, CameraData.create(self._compressed_video_paths[videoID])
        else:
            v = cv2.VideoCapture(self._compressed_video_paths[videoID])
            vframes = int(v.get(cv2.CAP_PROP_FRAME_COUNT))
            vlength = vframes/self.DATAPROPS.video.rate
            v.release()
            video_times = np.linspace(0,vlength,vframes)
            return GT, video_times
            
class UBFC2_Compressed(UBFC2):
    def __init__(self, compression_str, compression_origin="D:/Datasets/PPG/CompressedCopies/UBFC2", directory="D:/Datasets/PPG/UBFC2"):
        super().__init__(directory)
        self.name = "UBFC2_Compressed@" + compression_str
        compression_origin = os.path.join(compression_origin.replace('/', os.sep),compression_str)
        directory = directory.replace('/', os.sep)
        self._compressed_video_paths = [x.replace(directory, compression_origin)[:-3] + 'mp4' for x in self._video_paths]
    
    def get_video_path(videoID):
        return self._compressed_video_paths[videoID]
    
    def load_instance(self, videoID, include_video=True, clean_gt=True, gt_hr_params=None, **kwargs):
        start_path = os.path.dirname(self._video_paths[videoID])
        data_path = os.path.join(start_path, self.DATA_SUFFIX)

        
        with open(data_path, 'r') as f:
           x = f.readlines()

        bvp = x[0].split(' ')
        bvp = list(filter(lambda a: a != '', bvp))
        bvp = np.array(bvp).astype(np.float64)
        if clean_gt: bvp = medfilt(bvp)
        
        val_times = x[2].split(' ')
        val_times = list(filter(lambda a: a != '', val_times))
        val_times = np.array(val_times).astype(np.float64)
        #val_times = val_times - val_times[0]
        
        bvp = Signal(bvp, val_times)

        if gt_hr_params is None:
            hr = x[1].split(' ')
            hr = list(filter(lambda a: a != '', hr))
            hr = np.array(hr).astype(np.float64)
            if clean_gt: hr = medfilt(hr)
            hr = Signal(hr, val_times)
        else:
            hr = HR_from_BVP(bvp, **gt_hr_params)
            

        GT = SignalData(bvp=bvp, hr=hr)

        
        if include_video:
            return GT, CameraData.create(self._compressed_video_paths[videoID])
        else:
            v = cv2.VideoCapture(self._compressed_video_paths[videoID])
            vframes = int(v.get(cv2.CAP_PROP_FRAME_COUNT))
            vlength = vframes/v.get(cv2.CAP_PROP_FPS)
            v.release()
            video_times = np.linspace(0,vlength,vframes)
            return GT, video_times