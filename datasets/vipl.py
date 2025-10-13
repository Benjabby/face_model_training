# import numpy as np
# import os
# import csv
# import cv2
# import json
# from scipy.signal import medfilt
# from . import Dataset, SignalData, CameraData
# from ..signals import *

# class VIPL(Dataset):
    # """
    # """
    # name = 'VIPL'

    # DATAPROPS = {
        # 'video':{
            # 'ext':'mkv',
            # 'rate':25},
        # 'BVP':{
            # 'rate':1000},
        # 'ECG':None,
        # 'HR':{
            # 'rate':25},
        # 'O2':'indirect',
        # 'BR':{
            # 'rate':25},
    # }
    
    # COMPRESSED = True
    
    # DATA_SUFFIX = ".txt"
    
    
    # def __init__(self, directory="D:/Datasets/PPG/VIPL"):
        # super().__init__(directory)
        
        
    # def  get_instance_name(self, videoID):
        # name = os.path.splitext(os.path.basename(self._video_paths[videoID]))[0]
        # return name

    # def get_friendly_name(self, videoID):
        # name = self.get_instance_name(videoID)
        # num = int(name[1:4])
        # return name[0] + " " + str(num) + name[5:]

    # def load_instance(self, videoID, include_video=True, clean_gt=True, gt_hr_params=None, **kwargs):
        # base = self._video_paths[videoID]
        # name = self.get_instance_name(videoID)
        # bvp_path = os.path.join(os.path.dirname(os.path.dirname(base)),"Ground truth","BP_raw_1KHz",name.replace('_','-')+'-BP'+self.DATA_SUFFIX)
        
        # hr_br_path = os.path.join(os.path.dirname(os.path.dirname(base)),"Ground truth","Physiology",name+self.DATA_SUFFIX)
        
        # bvp = np.loadtxt(bvp_path)
        
        # with open(hr_br_path,'r') as f:
            # hr = f.readline()
            # br = f.readline()
        
        # hr = [x.strip() for x in hr.split(',')]
        # br = [x.strip() for x in br.split(',')]
        
        # hr = np.array(hr[2:],dtype=np.float64)
        # br = np.array(br[2:],dtype=np.float64)
        
        # end = len(hr)/self.DATAPROPS.video.rate
        
        # cut = int(len(bvp) - end*1000)
        
        # video_times = np.linspace(0, end, len(hr))
        # val_times = np.linspace(0, end, len(bvp)-cut)
        
        # bvp = bvp[:-cut]
        
        # if clean_gt:
            # bvp = medfilt(bvp)
            # hr = medfilt(hr)
            # br = medfilt(br)
            
        
        # bvp = Signal(bvp, val_times)
        # if gt_hr_params is None:
            # hr = Signal(hr, video_times)
        # else:
            # hr = HR_from_BVP(bvp, **gt_hr_params)
        # br = Signal(br, video_times)
        
        # GT = SignalData(bvp=bvp,hr=hr,br=br)

        # if include_video:
            # return GT, CameraData.create(self._video_paths[videoID], video_times)
        # else:
            # return GT, video_times
