########################################
# Code by Benjamin Tilbury             #
#       KTP Associate with UWS/Kibble  #
########################################

from abc import ABCMeta, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import warnings
import pandas as pd
from scipy.signal import stft, medfilt, butter, sosfiltfilt
from scipy.interpolate import interp1d, CubicSpline
from biosppy.signals import ecg as bioecg, resp as bioresp
from scipy.ndimage import uniform_filter, gaussian_filter1d
# import neurokit2 as nk

from ..consts import Globals

def _spectro(data, fs, win_size, stride, minHz, maxHz):
    segment = (win_size * fs)
    overlap = (fs*(win_size-stride))
    if segment>len(data):
        segment = len(data)
        overlap = 0
    nfft = max(2048, 60*2*fs)
    F, T, Z = stft(data, fs, nperseg=segment, noverlap=overlap, boundary='even', nfft=nfft)
    
    band = np.argwhere((F > minHz) & (F < maxHz)).flatten()
    spect = np.abs(Z[band, :])
    freqs = 60*F[band]
    
    vals = freqs[np.argmax(spect, axis=0)]

    return spect, freqs, T, vals

def HR_from_BVP(BVP, win_size=10, stride=1, **kwargs):
    fs = BVP.hz
    minHz = Globals.HR_MIN_HZ
    maxHz = Globals.HR_MAX_HZ
    _, _, times, hr = _spectro(BVP.data, fs, win_size, stride, minHz, maxHz)
    return Signal(hr, times)
    
def HR_from_ECG(ECG, **kwargs):
    out = bioecg.ecg(signal=ECG.data, sampling_rate=ECG.hz, show=False)
    return Signal(out['heart_rate'], out['heart_rate_ts'])
    #
    
def BR_from_BVP(BVP, smoothing=2, **kwargs):
    fs = BVP.hz
    bio_rsp = bioresp.resp(BVP.data, sampling_rate=fs,show=False)
    
    return Signal(bio_rsp['resp_rate'], bio_rsp['resp_rate_ts'])
    
def BR_from_ECG(ECG, **kwargs):
    pass

def O2_from_BVP(BVP, win_size=10, **kwargs):
    pass
    
def O2_from_ECG(ECG, **kwargs):
    pass

class PartialSignal():
    def __init__(self, data, hz, start_stop=None):
        self.data = data
        self.hz = hz
        self.start = None if start_stop is None else start_stop[0]
        self.stop = None if start_stop is None else start_stop[1]
        
    def __iter__(self):
        yield from [self.data, self.hz]

    def get_times(self):
        if self.hz==0:
            return np.zeros(self.data.shape)
        elif self.start is None:
            return np.linspace(0,len(self.data)*self.hz,len(self.data))
        else:
            return np.linspace(self.start,self.stop,len(self.data))



class WorkingSignal():

    def __init__(self, win_delay=0, buffer=None):#, init_data=None, init_times=None, init_proc_times=None):
        """
        Backed by lists for ease of adding.
        Args:
            win_delay : (float)
                []
            buffer : int or 'shedding'
                If 'shedding', this signal when get_window is called, this signal will discard all data before the start of the window.
                Only use when you can guarantee that any window taken from this signal will always have a beggining after or equal to the beggining of the previously taken window
            
        """
        
        # if (init_data is not None or init_times is not None or init_proc_times is not None):
            # if (init_data is None or init_times is None or init_proc_times is None):
                # raise ValueError("if initial data is specified, arrays for init_data, init_times, and init_proc_times must all be provided")
            # start = 0 if buffer is None else buffer
            # self.data = list(init_data[-start:])
            # self.times = list(init_times[-start:])
            # self.proc_times = list(init_proc_times[-start:])
            
        # else:
        self.data = []
        self.times = []
        self.proc_delays = []

        self.win_delay = win_delay
        self.buffer = buffer

    def __len__(self):
        return len(self.data)
    
    def convert(self):
        return PredictedSignal(self.data, self.times, self.win_delay, self.proc_delays)

    def append(self, data, t=None, proc_delay=0, default=0, use_prev=True):
        """
        Args
            data :
                Data to be 
            t : float
                The true timestamp associated with the data ignoring any delays.
                i.e. for data derived from a single frame, ref_time should be the timestamp of that frame.
                For data derived from a window, ref_time should be the timestamp in the MIDDLE of the window
            proc_delay : float     
                the time delay incured in the calculation/processing of this data 
            
        """
        
        if use_prev and len(self.data)>0 and (not np.isscalar(default) or default!='skip'):
            default = self.data[-1]
        
        if data is None:
            v = default
            if np.isscalar(default) and default=='skip':
                return
        elif np.isscalar(data):
            v = data if np.isfinite(data) else default
            if np.isscalar(v) and v=='skip':
                return
        elif np.isnan(data).any() and np.isscalar(default) and default=='skip':
            return
        elif np.isscalar(default):
            v = np.nan_to_num(data,nan=default)
        else:
            v = data
            nans = np.isnan(v)
            if nans.any():
                if v.shape!=default.shape:
                    raise RuntimeError("Default value must be the same shape as data or be scalar")
                v[np.isnan(v)] = default[np.isnan(v)]
                
        # breakpoint()
        if t is None:
            t = len(self.times)
            
        self.data.append(v)
        
        if len(self.times)>0 and t<self.times[-1]:
            raise RuntimeError("Timestamp provided is before the most recent value")
        
            
        self.times.append(t)
        self.proc_delays.append(proc_delay)
        
        # print("{}: {}".format(self,np.array(self.data).shape))
        # breakpoint()
        if self.buffer is not None and self.buffer!='shedding' and len(self.data)>self.buffer:
            self.data.pop(0)
            self.times.pop(0)
            self.proc_delays.pop(0)
        
        return self
        
        
    def temp_window(self, frames=360):
        x = np.array(self.times[-frames:])
        d = np.array(self.data[-frames:])
        
        if len(d)>2:
            fs = (len(d) - 1) / (t1 - t0)
            if force_fs is None:
                xn = np.linspace(t0, t1, len(d))
            else:
                vd = force_fs / fs
                xn = np.linspace(t0, t1, int(len(x)*vd))
            
            if d.dtype==np.dtype('O'):
                breakpoint()
            
            if d.ndim==1:
                interp = np.interp(xn, x, d)
            elif d.ndim==2:
                interp = np.stack([np.interp(xn,x,d[...,i]) for i in range(d.shape[-1])],axis=-1)
            else:
                sh = d.shape
                d = d.reshape(sh[0], -1)
                interp = np.stack([np.interp(xn,x,d[...,i]) for i in range(d.shape[-1])],axis=-1)
                interp = interp.reshape(sh)
        else:
            interp = d
            fs = 0
        
        if interp.ndim>2:
            interp = np.swapaxes(interp, 0, 1)
            # breakpoint()
        
        # print("window {}".format(interp.shape))
        return PartialSignal(interp, fs)
    
    # def overlap_add(self, )
    
    def get_raw_window(self, win_size, wait=False):
        if len(self.times)==0:
            if wait:
                return None
            else:
                return PartialSignal(np.array([]),0)
        
        if len(self.times)<win_size and wait:
            return None
        
        fs = 1.0/np.mean(np.diff(self.times[-win_size:]))
        t = self.times[-win_size:]
        return PartialSignal(np.array(self.data[-win_size:]), fs, [t[0], t[-1]])
    
    def get_window(self, win_seconds=None, n_frames=None, start_time=None, wait=False, force_size=None, force_fs=None):
        if force_size is not None and force_fs is not None:
            raise ValueError("Cannot use force_size and force_fs at the same time")
            
        arg_v = int(win_seconds is not None) + int(n_frames is not None) + int(start_time is not None)
        if arg_v>1:
            raise ValueError("Only one window selection criteria of  'win_seconds', 'n_frames' or 'start_time' should be specified")
        elif arg_v==0:
            raise ValueError("One of either 'win_seconds', 'n_frames' or 'start_time', must be specified")
            
        
        if len(self.times)==0:
            if wait:
                return None
            else:
                return PartialSignal(np.array([]),0)
        
        x = np.array(self.times)
        t1 = x[-1]
        if n_frames is None:
            
            if win_seconds=='all':
                i0 = 0
            else:
                if start_time is None:
                    t0 = t1-win_seconds
                else:
                    t0 = start_time
                i0 = np.searchsorted(x, t0)-1
                if i0<0:
                    if wait:
                        return None
                    i0 = 0
        else:
            if n_frames>len(self.times) and wait:
                return None
                
            i0 = -n_frames
        
        x = x[i0:]
        t0 = x[0]
        d = np.array(self.data[i0:])
        
        
        # TODO replace this as a an argument instead of attribute of Signal
        if self.buffer=='shedding':
            self.times = self.times[i0:]
            self.data = self.data[i0:]
            self.proc_delays = self.proc_delays[i0:]
        elif len(d)==self.buffer:
            warnings.warn("Buffer size too small for windowing",stacklevel=2)
        
        if len(d)>2:
            sz = len(d) if force_size is None else force_size
            fs = (sz - 1) / (t1 - t0)
            if force_fs is None:
                xn = np.linspace(t0, t1, len(d))
            else:
                vd = force_fs / fs
                xn = np.linspace(t0, t1, int(len(x)*vd))
            
            if d.dtype==np.dtype('O'):
                breakpoint()
            
            try:
                if d.ndim==1:
                    interp = np.interp(xn, x, d)
                elif d.ndim==2:
                    interp = np.stack([np.interp(xn,x,d[...,i]) for i in range(d.shape[-1])],axis=-1)
                else:
                    sh = d.shape
                    d = d.reshape(sh[0], -1)
                    interp = np.stack([np.interp(xn,x,d[...,i]) for i in range(d.shape[-1])],axis=-1)
                    interp = interp.reshape(sh)
            except Exception as e:
                breakpoint()
        else:
            interp = d
            fs = 0
        
        if interp.ndim>2:
            interp = np.swapaxes(interp, 0, 1)
            # breakpoint()
        
        # print("window {}".format(interp.shape))
        return PartialSignal(interp, fs, [t0, t1])

    # def raw_window(self, window_size=256, fs=None):
        # d = np.array(self.data[-window_size:])
        # if fs is None:
            # if len(d)>2:
                # a = self.out_times[-min(window_size,len(self.out_times))]
                # b = self.out_times[-1]
                # fs = (len(d) - 1) / (b - a)
            # else:
                # fs = 0
                
        # #print(d.shape)
        # if d.ndim>2:
            # d = np.swapaxes(d, 0, 1)
        # #print(d.shape)
        # return PartialSignal(d,fs)
    
    # def other_window(self, window_size=256, fs=30):
        # d = np.array(self.data[-window_size:])
        # x = np.array(self.out_times[-window_size:])
        # a = x[0]
        # b = x[-1]
        # xn = np.linspace(a,b,len(x))
        # if d.ndim==1:
            # return PartialSignal(np.interp(xn,x,d), fs)
        # else:
            # return PartialSignal(np.stack([np.interp(xn,x,d[...,i]) for i in range(d.shape[-1])],axis=-1),fs)

    # def heartwave_window(self, window_size=256):
        # d = np.array(self.data[-window_size:])
        # # print("window {}".format(d.shape))
        # x = np.array(self.out_times[-window_size:])
        # if len(d)>2:
            # t0 = x[0]
            # t1 = x[-1]
            # xn = np.linspace(t0, t1, len(d))
            # fs = (len(d) - 1) / (t1 - t0)
            
            # if d.dtype==np.dtype('O'):
                # breakpoint()
            
            # if d.ndim==1:
                # interp = np.interp(xn, x, d)
            # elif d.ndim==2:
                # interp = np.stack([np.interp(xn,x,d[...,i]) for i in range(d.shape[-1])],axis=-1)
            # else:
                # sh = d.shape
                # d = d.reshape(sh[0], -1)
                # interp = np.stack([np.interp(xn,x,d[...,i]) for i in range(d.shape[-1])],axis=-1)
                # interp = interp.reshape(sh)
        # else:
            # interp = d
            # fs = 0
        
        # if interp.ndim>2:
            # interp = np.swapaxes(interp, 0, 1)
            # # breakpoint()
        
        # # print("window {}".format(interp.shape))
            
        # return PartialSignal(interp, fs)


    # def alt_window(self, window_size=256, fs=30):
        # d = np.array(self.data[-self._MAX_PROC:])
        # x = np.array(self.out_times[-self._MAX_PROC:])
        # b = x[-1]
        # t = window_size/fs
        # xn = np.linspace(b-t+1/fs,b,window_size)
        # if d.ndim==1:
            # return PartialSignal(np.interp(xn,x,d), fs)
        # else:
            # return PartialSignal(np.stack([np.interp(xn,x,d[...,i]) for i in range(d.shape[-1])],axis=-1),fs)
        

    # def get_window(self, window_size=256):
        # d = np.array(self.data[-window_size:])
        # x = np.array(self.out_times[-window_size:])
        # a = x[0]
        # b = x[-1]
        ## xn, sp = np.linspace(a,b,min(window_size,len(self.out_times)),retstep=True)
        # xn, sp = np.linspace(a,b,window_size,retstep=True) 
        # if d.ndim==1:
            # return PartialSignal(np.interp(xn,x,d), 1/sp)
        # else:
            # return PartialSignal(np.stack([np.interp(xn,x,d[...,i]) for i in range(d.shape[-1])],axis=-1),1/sp)
            
            

class Signal():
    
    @staticmethod
    def align_signals(A, B, times_to_use, method='previous', ignore_win_delay=False, ignore_proc_delay=False):
        if isinstance(times_to_use, str):
            if times_to_use == 'A':
                times = A.get_times(ignore_win_delay=ignore_win_delay, ignore_proc_delay=ignore_proc_delay)
                a = A.data
                b = B.resample(times, method=method, ignore_win_delay=ignore_win_delay, ignore_proc_delay=ignore_proc_delay)
            if times_to_use == 'B':
                times = B.get_times(ignore_win_delay=ignore_win_delay, ignore_proc_delay=ignore_proc_delay)
                a = A.resample(times, method=method, ignore_win_delay=ignore_win_delay, ignore_proc_delay=ignore_proc_delay)
                b = B.data
        elif isinstance(times_to_use, np.ndarray) or isinstance(times_to_use, list):
            times = times_to_use
            a = A.resample(times, method=method, ignore_win_delay=ignore_win_delay, ignore_proc_delay=ignore_proc_delay)
            b = B.resample(times, method=method, ignore_win_delay=ignore_win_delay, ignore_proc_delay=ignore_proc_delay)
        else:
            raise RuntimeError("'times_to_use' must be a string of 'A' or 'B', or an array/list of timestamps")
        
        return a, b, times
    
    def __init__(self, data, sampling):
        """
        Args:
            data        : array
                The signal data
            sampling    : array or int
                Either an array of sampling times, or the sampling rate in hertz
        """
        
        self.data = data if isinstance(data, np.ndarray) else np.array(data)
        
        if np.isscalar(sampling):
            self.hz = sampling
            self.int_hz = np.round(sampling)

            self.times = np.arange(len(data))/sampling
            self.sampling = 'uniform'
            self.hz = sampling
        else:
            self.sampling = 'timestamped'
            self.times = sampling if isinstance(sampling, np.ndarray) else np.array(sampling)
            self.hz = 1.0/np.mean(np.diff(sampling))
            self.int_hz = np.round(self.hz)

            if len(self.times)!=len(self.data):
                raise RuntimeError("Signal timestamps and data must be the same length")
        
        self.start_offset = self.times[0]

    def resample(self, timestamps, method='previous', **kwargs):
        times = self.get_times(**kwargs)
        f = interp1d(self.times, self.data, kind=method, fill_value=(self.data[0],self.data[-1]), bounds_error=False, assume_sorted=True)
        return f(timestamps)
        
    def get_uniform(self, spline=True, data_only=False, int_hz=False):
        if self.sampling == 'uniform':
            if data_only:
                return self.data
            else:
                return self
        else:
            if int_hz:
                newhz = np.round(self.hz)
                newt = np.arange(len(self.data))/newhz + self.times[0]
            else:
                newt, st = np.linspace(self.times[0], self.times[-1], len(self.times), retstep=True)
                newhz = 1/st
            
            if spline:
                interpolator = CubicSpline(self.times, self.data)
            else:
                interpolator = interp1d(self.times, self.data, kind='linear', fill_value='extrapolate', bounds_error=False, assume_sorted=True)
            
            new = interpolator(newt)
            
            if data_only:
                return new
            else:
                return Signal(new, newt)
    
    # def get_filtered(self, filt):
        
    def get_times(self, **kwargs):
        return self.times
        
    

    # def plot(self,show=True):
        # plt.plot(self.times, self.data)
        # if show:
            # plt.show()
            
    # def __iter__(self):
        # yield from [self.data, self.times]


class PredictedSignal(Signal):
    def __init__(self, data, times, win_delay, proc_delays):
        super().__init__(data, times)
        self.win_delay = win_delay if win_delay is not None else 0
        if proc_delays is None:
            self.proc_delays = 0
        else:
            self.proc_delays = proc_delays if isinstance(proc_delays, np.ndarray) else np.array(proc_delays)
            if self.proc_delays.size == 0:
                self.proc_delays = 0
            elif self.proc_delays.size == 1:
                self.proc_delays = np.asscalar(self.proc_delays)
            elif len(proc_delays)!=len(times):
                raise ValueError("Processing delays must be the same length as timestamps or None")
            
        
        
    # def __call__(self, ignore_win_delay=False, ignore_proc_delay=False):
        # yield from [self.data, self.times]
        
    def get_times(self, ignore_win_delay=False, ignore_proc_delay=False, **kwargs):
        times = self.times
        
        if not ignore_win_delay:
            times = times + self.win_delay
            
        if not ignore_proc_delay:
            times = times + self.proc_delays
            
        return times