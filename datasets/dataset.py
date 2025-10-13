from abc import ABCMeta, abstractmethod
import os
from attrdict import AttrDict
from ..signals import Signal, HR_from_BVP, HR_from_ECG, BR_from_BVP, BR_from_ECG, O2_from_BVP, O2_from_ECG
import cv2
import numpy as np
import warnings
import hashlib
from sklearn.model_selection import train_test_split
import stat
import ffmpeg
import glob
import subprocess

WARN = False
TRAIN_SPLIT = 0.75

MJPEG_QUALITY = {'high':2,'medium':16,'low':30}
H264_QUALITY = {'high':15,'medium':23,'low':31}
H265_QUALITY = {'high':20,'medium':28,'low':36}

##def infer_missing(GT, target_times : dict):
##    types = target.keys()
##    if 'HR' not in types:
##        
##        
##    return {
##        'BVP':None,
##        'ECG':None,
##        'HR':None,
##        'O2':None,
##        'BR':None}

class SignalData():
    def __init__(self, bvp:Signal=None, ecg:Signal=None, hr:Signal=None, br:Signal=None, o2:Signal=None):
        self.BVP = bvp
        self.ECG = ecg
        self.HR = hr
        self.BR = br
        self.O2 = o2
        
    # def fill_missing(self, **kwargs):
        # if self.BVP is None and self.ECG is None:
            # warnings.warn("Signal data contains no BVP or ECG to calculate indirect values")
            # return
        
        # use_bvp = self.BVP is not None
        
        # if self.BR is None:
            # br_paramss = kwargs['br_paramss'] if 'br_paramss' in kwargs else {}
            # self.BR = BR_from_BVP(self.BVP, **br_paramss) if use_bvp else BR_from_ECG(self.ECG, **br_paramss)
        
        # if self.O2 is None:
            # o2_params = kwargs['o2_params'] if 'o2_params' in kwargs else {}
            # self.O2 = O2_from_BVP(self.BVP, **o2_params) if use_bvp else O2_from_ECG(self.ECG, **o2_params)
        


# TODO MAYBE This should really be a context manager
class CameraData(metaclass=ABCMeta):
    
    @staticmethod
    def create(path, timestamps=None):
        if os.path.isfile(path):
            return _CameraDataVideo(path, timestamps=timestamps)
        else:
            return _CameraDataFrames(path, timestamps=timestamps)
    
    @abstractmethod
    def __init__(self, path, timestamps=None):
        pass
        
    
    @property
    @abstractmethod
    def width(self):
        pass
        
    @property
    @abstractmethod
    def height(self):
        pass
    
    @property
    @abstractmethod
    def fps(self):
        pass
    
    @property
    @abstractmethod
    def nframes(self):
        pass
    
    @property
    @abstractmethod
    def times(self):
        pass
    
    def __iter__(self):
        return self
    
    @abstractmethod
    def __next__(self):
        pass
        
    @abstractmethod
    def close(self):
        pass
        

class _CameraDataVideo(CameraData):
    
    def __init__(self, path, timestamps=None):
        assert os.path.isfile(path), "_CameraDataVideo must be instantiated with a video file, not a directory. CameraData.create is preferred"
        self._path = path
        self._cam = cv2.VideoCapture(path)
        self._width = int(self._cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._fps = self._cam.get(cv2.CAP_PROP_FPS)
        self._frame = np.zeros((self._height,self._width,3),dtype=np.uint8)
        self._nframes = int(self._cam.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.curr_frame = 0
        
        if timestamps is None:
            vlength = self.nframes/self.fps
            self._times = np.linspace(0,vlength,self.nframes)
        elif len(timestamps)<self.nframes:
            raise RuntimeError("Timestamps provided for for video at '{}' do not contain enough entries for the frames of the video file ({} timestamps for {} frames)".format(path,len(timestamps),self.nframes))
        elif len(timestamps)>self.nframes:
            if WARN:
                warnings.warn("Timestamps provided for video at '{}' contain more entries than frames of the video file ({} timestamps for {} frames)\nThese extra timestamps will be ignored".format(path,len(timestamps),self.nframes),stacklevel=2)
            self._times = timestamps[:self.nframes]
        else:
            self._times = timestamps
            
    @property
    def width(self):
        return self._width
    @property
    def height(self):
        return self._height
    @property
    def fps(self):
        return self._fps
    @property
    def nframes(self):
        return self._nframes
    @property
    def times(self):
        return self._times
        
    def reset(self):
        self.curr_frame = 0
        self._cam = self.cv2.VideoCapture(path)
        
    def __next__(self):
        okay, _ = self._cam.read(self._frame)
        if not okay:
            self._cam.release()
            raise StopIteration
        else:
            t = self.times[self.curr_frame]
            self.curr_frame += 1
            return self._frame, t
    
    def close(self):
        self._cam.release()
    
class _CameraDataFrames(CameraData):
    
    def __init__(self, path, timestamps=None):
        assert os.path.isdir(path), "_CameraDataFrames must be instantiated with a directory of frames, not a file. CameraData.create is preferred"
        self._path = path
        self._file_names = sorted(os.listdir(path))
        proto = cv2.imread(os.path.join(path,self._file_names[0]))
        self._height, self._width, _ = proto.shape
        self._nframes = len(self._file_names)
        
        self.curr_frame = 0
        
        assert timestamps is not None, "_CameraDataFrames must be instantiated with timestamps"
        if len(timestamps)<self.nframes:
            raise RuntimeError("Timestamps provided for for video at '{}' do not contain enough entries for the frames of the video file ({} timestamps for {} frames)".format(path,len(timestamps),self.nframes))
        elif len(timestamps)>self.nframes:
            warnings.warn("Timestamps provided for video at '{}' contain more entries than frames of the video file ({} timestamps for {} frames)\nThese extra timestamps will be ignored".format(path,len(timestamps),self.nframes),stacklevel=2)
            self._times = timestamps[:self.nframes]
        else:
            self._times = timestamps
        
        self._fps = 1/np.mean(np.diff(self.times))
        
        
    @property
    def width(self):
        return self._width
    @property
    def height(self):
        return self._height
    @property
    def fps(self):
        return self._fps
    @property
    def nframes(self):
        return self._nframes
    @property
    def times(self):
        return self._times
    
    def reset(self):
        self.curr_frame = 0
    
    def __next__(self):
        if self.curr_frame>=self.nframes:
            raise StopIteration
        else:
            frame = cv2.imread(os.path.join(self._path,self._file_names[self.curr_frame]))
            t = self.times[self.curr_frame]
            self.curr_frame += 1
            return frame, t
            
            
    def close(self):
        pass

# class DatasetGenerator:
    ####TODO

class Dataset(metaclass=ABCMeta):
    """
    This is the abstract class used for creating a new Dataset Class.
    Heavily modified version of Dataset from pyVHR.datasets.dataset to include timestamps and other ground truth signals
    """

    DATAPROPS = {
        'video':None,
        'BVP':None,
        'ECG':None,
        'HR':None,
        'O2':None,
        'BR':None,
    }
    
    def __len__(self):
        return self.size
    
    def __init__(self, directory=None, warn=True):
        """
        Args:
            directory (str): path of the dataset
        """
        # -- load filenames
        self._video_paths = []  # list of all video filenames
        self._invalid_paths = {}
        self.size = 0        # num of videos in the dataset
        self.directory = directory.replace('/', os.sep)
        self.DATAPROPS = AttrDict(self.DATAPROPS)
        
        self._init_filenames()
        
        # -- number of videos
        self.size = len(self._video_paths)
        
        if len(self._invalid_paths)>0 and warn:
            warnings.warn("{} contains {} invalid ground truth files\n".format(self.name, len(self._invalid_paths)) + "\n".join("{}: {}".format(self.get_instance_name(videoPath=k), v) for k, v in self._invalid_paths.items()) + "\nThese will be ignored",stacklevel=2)
        
        base = os.path.abspath(self.directory)
        sorted(self._video_paths, key=lambda i: os.path.relpath(i, base))
        
        rstate = abs(hash(self.name)) % (2**32-1) # Because scikit-learn is a little bitch baby and can't handle proper 64 bit signed integers as random seeds.
        self.trainID, self.testID = train_test_split(np.arange(self.size), train_size=TRAIN_SPLIT, random_state=rstate)
        
    def create_compressed(self, output_dir, method='mjpeg', level='high', chroma='444'):
        # TODO h264
        # TODO h265
        # TODO chroma
        assert "Compressed" not in self.__class__.__name__, "Dataset is already compressed"
        if self.DATAPROPS.video.ext in ['png','jpeg','jpg','gif','bmp']:
            frame_rates = []
            for i in range(self.size):
                _, vidobj = self.load_instance(i, include_video=True)
                frame_rates.append(vidobj.fps)
            Dataset._create_compressed_from_images(self.name, self.directory, output_dir, self._video_paths, frame_rates, method, level, chroma)
        else:
            Dataset._create_compressed_from_video(self.name, self.directory, output_dir, self._video_paths, method, level, chroma)
            
    @staticmethod
    def _create_compressed_from_video(name, input_dir, output_dir, video_paths, method, level, chroma, verbose=0):
        output_dir = os.path.join(output_dir, name)
        ext = '-'.join([method, level, chroma])
        output_dir = os.path.join(output_dir, ext)
        os.makedirs(output_dir, exist_ok=True)
        
        if verbose>0: print("Making {} version of {}".format(ext, name))
        
        for i,video_path in enumerate(video_paths):
            
            new_path = video_path.replace(input_dir, output_dir)[:-3] + 'mp4'
            
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            if os.path.isfile(new_path):
                if verbose>0: print("{} Already exists, skipping...".format(new_path))
                continue
            
            if method=='mjpeg':
                q = MJPEG_QUALITY.get(level,2)
                ffmpeg.input(video_path).output(new_path, vcodec='mjpeg',qscale=q).global_args('-loglevel', 'error').run(overwrite_output=True)
            elif method=='h264':
                q = H264_QUALITY.get(level,15)
                ffmpeg.input(video_path).output(new_path, vcodec='libx264',crf=q).global_args('-loglevel', 'error').run(overwrite_output=True)
            else: #method=='h265':
                q = H265_QUALITY.get(level,20)
                ffmpeg.input(video_path).output(new_path, vcodec='libx265',crf=q).global_args('-loglevel', 'error').run(overwrite_output=True)
            
            
            if verbose>0: print("{} of {} done".format(i+1, len(video_paths)))
        
    @staticmethod
    def _create_compressed_from_images(name, input_dir, output_dir, image_paths, frame_rates, method, level, chroma, verbose=0):
        output_dir = os.path.join(output_dir, name)
        ext = '-'.join([method, level, chroma])
        output_dir = os.path.join(output_dir, ext)
        os.makedirs(output_dir, exist_ok=True)
        
        if verbose>0: print("Making {} version of {}".format(ext, name))
        
        
        
        for i,(folder,fps) in enumerate(zip(image_paths,frame_rates)):
            
            new_path = folder.replace(input_dir, output_dir) + ".mp4"
            
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            if os.path.isfile(new_path):
                if verbose>0: print("{} Already exists, skipping...".format(new_path))
                continue
            
            filenames = sorted(glob.glob(os.path.join(folder,'*.png')), key=lambda x: int(os.path.basename(x).split('.')[0].removeprefix("Image")))
            tmppath = os.path.join(os.path.dirname(new_path),"tmp.txt")
            with open(tmppath,'wb') as outfile:
                for filename in filenames:
                    outfile.write(f"file '{filename}'\n".encode())
                    outfile.write(f"duration 0.5\n".encode())
            
            if method=='mjpeg':
                q = MJPEG_QUALITY.get(level,2)
                
                command_line = f"ffmpeg -y -r {fps} -f concat -safe 0 -i {tmppath} -qscale {q} -vcodec mjpeg {new_path} -loglevel error"
                
            elif method=='h264':
                q = H264_QUALITY.get(level,15)
                command_line = f"ffmpeg -y -r {fps} -f concat -safe 0 -i {tmppath} -crf {q} -vcodec libx264 {new_path} -loglevel error"
            else: # method=='h265':
                q = H265_QUALITY.get(level,20)
                command_line = f"ffmpeg -y -r {fps} -f concat -safe 0 -i {tmppath} -crf {q} -vcodec libx265 {new_path} -loglevel error"
            
            pipe = subprocess.Popen(command_line, shell=True, stdout=subprocess.PIPE).stdout
            output = pipe.read().decode()
            pipe.close()
            
            # breakpoint()
            os.remove(tmppath)
            if verbose>0: print("{} of {} done".format(i+1, len(image_paths)))
    
    
    def _init_filenames(self):
        for root, dirs, files in os.walk(self.directory):
            for f in files:
                filename = os.path.join(root, f)
                path, name = os.path.split(filename)

                # -- select video
                if filename.endswith(self.DATAPROPS.video.ext): #and (name.find(self.VIDEO_SUBSTRING) >= 0):
                    fpath = os.path.abspath(filename)
                    validity = self._get_gt_validity(fpath)
                    if validity.upper() == "GOOD":
                        self._video_paths.append(fpath)
                    else:
                        self._invalid_paths[fpath] = validity

                # -- select signal
                #if filename.endswith(self.SIG_EXT) and (name.find(self.SIG_SUBSTRING) >= 0):
                    #self.sigFilenames.append(filename)

    
    def get_video_path(self, videoID):
        """Get video file name (or folder containing frame images) given the index."""
        return self._video_paths[videoID]
        
    @abstractmethod
    def get_instance_name(self, videoID=0, videoPath=None):
        pass
        
    
    @abstractmethod
    def get_friendly_name(self, videoID=0, videoPath=None):
        pass
        
        
    @abstractmethod
    def _get_gt_validity(self, videoPath):
        pass

    # @abstractmethod
    # def load_ground_truth(self, videoID):
        # pass
    
    # @abstractmethod
    # def load_video(self, videoID, **kwargs):
        # pass

    @abstractmethod
    def load_instance(self, videoID, include_video=True, clean_gt=True, gt_hr_params=None, **kwargs):
        pass
        

    def has(self, prop):
        return bool(self.DATAPROPS[prop])

    def isGT(self, prop):
        return self.has(prop) and self.DATAPROPS[prop]!='indirect'
