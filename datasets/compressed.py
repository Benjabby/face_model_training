# from . import Dataset

# There is a better way to do the compressed thing than having individual classes. Just got to figure out how to either on the fly subclass or use a wrapper 
# class Compressed(Dataset):
    
    
    # def load_instance(self, videoID, include_video=True, clean_gt=True, gt_hr_params=None, **kwargs): 
        # GT, video_times = super().load_instance(videoID, include_video=False, clean_gt=clean_gt, gt_hr_params=gt_hr_params, **kwargs)

        # if include_video:
            # return GT, CameraData.create(self._compressed_video_paths[videoID], video_times)
        # else:
            # return GT, video_times