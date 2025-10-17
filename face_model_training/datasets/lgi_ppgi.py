import xml.etree.ElementTree as ET
import numpy as np
import os
import warnings
from scipy.signal import medfilt

from . import Dataset, SignalData, CameraData
from ..signals import *

# TODO seperate categories

class LGI_PPGI(Dataset):
    """
    LGI-PPGI Dataset

    .. LGI-PPGI dataset structure:
    .. ---------------------------
    ..    datasetDIR/
    ..    |
    ..    |-- vidDIR1/
    ..    |   |-- videoFile1.avi
    ..    |
    ..    |...
    ..    |
    ..    |-- vidDIRM/
    ..        |-- videoFile1.avi
    """
    name = 'LGI_PPGI'

    DATAPROPS = {
        'video':{
            'ext':'avi',
            'rate':'timestamp',
            'compressed':False},
        'BVP':{
            'rate':60},
        'ECG':None,
        'HR':{
            'rate':60},
        'O2':'indirect',
        'BR':'indirect',
    }
    

    DATA_SUFFIX = "cms50_stream_handler.xml"
    TIME_SUFFIX = "cv_camera_sensor_timer_stream_handler.xml"
    
    def __init__(self, directory="D:/Datasets/PPG/LGI-PPGI"):
        super().__init__(directory)
        
    def get_instance_name(self, videoID=0, videoPath=None):
        videoPath = self._video_paths[videoID] if videoPath is None else videoPath
        _, name = os.path.split(os.path.dirname(videoPath))
        return name

    def get_friendly_name(self,videoID=0, videoPath=None):
        videoPath = self._video_paths[videoID] if videoPath is None else videoPath
        name = self.get_instance_name(videoPath=videoPath)
        name = name.replace('_', ' ')
        name = name.title()
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
        tree = ET.parse(os.path.join(start_path, self.DATA_SUFFIX))
        
        bvp_elements = tree.findall('.//*/value2')
        bvp = [int(item.text) for item in bvp_elements]
        
        hr_elements = tree.findall('.//*/value1')
        hr = [int(item.text) for item in hr_elements]


        # I'm just going to have to trust this method used by pyVHR for syncing the two signals
        # The options are:
        # [ ] 1: Assume the oxymeter sampling was not exactly 60hz, but started and stopped exactly when the video did  -> leave signal as-is and calculate new Hz
        # [x] 2: Assume the oxymeter is exactly 60hz, stopped exactly when the video did, but started slightly early    -> remove samples from the start
        # [ ] 3: Assume the oxymeter is exactly 60hz, started exactly when the video did, but stopped slightly late     -> remove samples from the end
        # Personally I would do the first one, but I'm going to trust that the pyVHR people know more than I do and stick with 2
    
        vid_xml_filename = os.path.join(start_path, self.TIME_SUFFIX)
        tree = ET.parse(vid_xml_filename)
        video_times = tree.findall('.//*/value1')
        video_times = [float(item.text) for item in video_times]
        last_vid_time = video_times[-1]
        video_times = np.array(video_times)/1000
        
        # There's no real reason to do these seperately tbh
        n_bvp_samples = len(bvp)
        last_bvp_time = int((n_bvp_samples*1000)/self.DATAPROPS.BVP.rate)
        n_hr_samples = len(hr)
        last_hr_time = int((n_hr_samples*1000)/self.DATAPROPS.HR.rate)

        diff_bvp = ((last_bvp_time - last_vid_time)/1000)
        diff_hr = ((last_hr_time - last_vid_time)/1000)

        diff_samples_bvp =  max(0,round(diff_bvp*self.DATAPROPS.BVP.rate))
        diff_samples_hr =  max(0,round(diff_hr*self.DATAPROPS.HR.rate))
        
        bvp = np.array(bvp[diff_samples_bvp:])
        hr = np.array(hr[diff_samples_hr:])

        if clean_gt:
            bvp = medfilt(bvp)
            hr = medfilt(hr)

        bvp = Signal(bvp, self.DATAPROPS.BVP.rate)
        
        if gt_hr_params is None:
            hr = Signal(hr, self.DATAPROPS.HR.rate)
        else:
            hr = HR_from_BVP(bvp, **gt_hr_params)
            
        GT = SignalData(bvp=bvp,hr=hr)

        if include_video:
            return GT, CameraData.create(self._video_paths[videoID], video_times)
        else:
            return GT, video_times
    
    
class LGI_PPGI_Compressed(LGI_PPGI):
    def __init__(self, compression_str, compression_origin="D:/Datasets/PPG/CompressedCopies/LGI-PPGI", directory="D:/Datasets/PPG/LGI-PPGI"):
        super().__init__(directory)
        self.name = "LGI_PPGI_Compressed@" + compression_str
        compression_origin = os.path.join(compression_origin.replace('/', os.sep),compression_str)
        directory = directory.replace('/', os.sep)
        self._compressed_video_paths = [x.replace(directory, compression_origin)[:-3] + 'mp4' for x in self._video_paths]
    
    def load_instance(self, videoID, include_video=True, clean_gt=True, gt_hr_params=None, **kwargs): 
        start_path = os.path.dirname(self._video_paths[videoID])
        tree = ET.parse(os.path.join(start_path, self.DATA_SUFFIX))
        
        bvp_elements = tree.findall('.//*/value2')
        bvp = [int(item.text) for item in bvp_elements]
        
        hr_elements = tree.findall('.//*/value1')
        hr = [int(item.text) for item in hr_elements]


        # I'm just going to have to trust this method used by pyVHR for syncing the two signals
        # The options are:
        # [ ] 1: Assume the oxymeter sampling was not exactly 60hz, but started and stopped exactly when the video did  -> leave signal as-is and calculate new Hz
        # [x] 2: Assume the oxymeter is exactly 60hz, stopped exactly when the video did, but started slightly early    -> remove samples from the start
        # [ ] 3: Assume the oxymeter is exactly 60hz, started exactly when the video did, but stopped slightly late     -> remove samples from the end
        # Personally I would do the first one, but I'm going to trust that the pyVHR people know more than I do and stick with 2
    
        vid_xml_filename = os.path.join(start_path, self.TIME_SUFFIX)
        tree = ET.parse(vid_xml_filename)
        video_times = tree.findall('.//*/value1')
        video_times = [float(item.text) for item in video_times]
        last_vid_time = video_times[-1]
        video_times = np.array(video_times)/1000
        
        # There's no real reason to do these seperately tbh
        n_bvp_samples = len(bvp)
        last_bvp_time = int((n_bvp_samples*1000)/self.DATAPROPS.BVP.rate)
        n_hr_samples = len(hr)
        last_hr_time = int((n_hr_samples*1000)/self.DATAPROPS.HR.rate)

        diff_bvp = ((last_bvp_time - last_vid_time)/1000)
        diff_hr = ((last_hr_time - last_vid_time)/1000)

        diff_samples_bvp =  max(0,round(diff_bvp*self.DATAPROPS.BVP.rate))
        diff_samples_hr =  max(0,round(diff_hr*self.DATAPROPS.HR.rate))
        
        bvp = np.array(bvp[diff_samples_bvp:])
        hr = np.array(hr[diff_samples_hr:])

        if clean_gt:
            bvp = medfilt(bvp)
            hr = medfilt(hr)

        bvp = Signal(bvp, self.DATAPROPS.BVP.rate)
        
        if gt_hr_params is None:
            hr = Signal(hr, self.DATAPROPS.HR.rate)
        else:
            hr = HR_from_BVP(bvp, **gt_hr_params)
            
        GT = SignalData(bvp=bvp,hr=hr)

        if include_video:
            return GT, CameraData.create(self._compressed_video_paths[videoID], video_times)
        else:
            return GT, video_times