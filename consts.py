import numpy as np
from dataclasses import dataclass, field
import os
import json
import hashlib


class Globals:
    """
        See default_global_constants.json
    """
    
    DEFAULT_PREV = None
    DEFAULT_COL_TYPE = None
    DEFAULT_HR = None
    
    HR_MIN_HZ = None
    HR_MAX_HZ = None
    HR_MIN = None
    HR_MAX = None
    
    BR_MIN_HZ = None
    BR_MAX_HZ = None
    BR_MIN = None
    BR_MAX = None
    
    RPPG_MIN_HZ = None
    RPPG_MAX_HZ = None
    
    @staticmethod
    def get_json_string():
        d = {k:v for k,v in Globals.__dict__.items() if k not in ['get_hash','get_json_string','__module__','__doc__','__dict__','__weakref__']}
        return json.dumps(d, separators=(',', ':'),sort_keys=True)
        
    @staticmethod
    def get_hash():
        string = Globals.get_json_string()
        return hashlib.md5(string.encode('utf-8')).hexdigest()
        
filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),"default_global_constants.json")

with open(filename) as json_file:
    js = json.load(json_file)

for key, value in js.items():
    if hasattr(Globals, key):
        if key not in ['DEFAULT_COL_TYPE','DEFAULT_PREV']:
            value = float(value)
        setattr(Globals, key, value)

setattr(Globals, 'HR_MIN', Globals.HR_MIN_HZ * 60)
setattr(Globals, 'HR_MAX', Globals.HR_MAX_HZ * 60)
setattr(Globals, 'BR_MIN', Globals.BR_MIN_HZ * 60)
setattr(Globals, 'BR_MAX', Globals.BR_MAX_HZ * 60)

setattr(Globals, 'RPPG_MIN_HZ', min(Globals.HR_MIN, Globals.BR_MIN))
setattr(Globals, 'RPPG_MAX_HZ', max(Globals.HR_MAX, Globals.BR_MAX))


class Landmarks():
    LEFT_EYE = [157,144, 145, 22, 23, 25, 154, 31, 160, 33, 46, 52, 53, 55, 56, 189, 190, 63, 65, 66, 70, 221, 222, 223, 225, 226, 228, 229, 230, 231, 232, 105, 233, 107, 243, 124]
    RIGHT_EYE = [384, 385, 386, 259, 388, 261, 265, 398, 276, 282, 283, 285, 413, 293, 296, 300, 441, 442, 445, 446, 449, 451, 334, 463, 336, 464, 467, 339, 341, 342, 353, 381, 373, 249, 253, 255]
    MOUTH = [391, 393, 11, 269, 270, 271, 287, 164, 165, 37, 167, 40, 43, 181, 313, 314, 186, 57, 315, 61, 321, 73, 76, 335, 83, 85, 90, 106]
    
    LEFT_CHEEK = [93, 177, 213, 192, 214, 212, 186, 92, 165, 203, 36, 101, 118, 117, 116, 34, 162, 127, 234, 227, 137, 123, 147, 187, 50, 207, 205, 206, 216]
    RIGHT_CHEEK = [391,322, 410, 432, 434, 416, 433, 401, 323, 454, 356, 389, 264, 345, 347, 330, 266, 423, 426, 425, 280, 436, 427, 411, 376, 366, 352, 447, 346]
    
    NOSE = [6, 197, 195, 5, 4, 1, 19, 94, 370, 354, 274, 275, 281, 248, 419, 351, 412, 399, 456, 363, 440, 457, 461, 462, 250, 458, 459, 309, 438, 344, 360, 420, 437, 343, 122, 196, 3, 51, 45, 44, 125, 141, 241, 237, 220, 134, 236, 174, 188, 114, 217, 198, 131, 115, 218, 79, 239, 238, 20, 242]
    FOREHEAD = [54, 68, 63, 105, 104, 103, 67, 69, 66, 107, 108, 109, 10, 151, 9, 336, 337, 338, 296, 299, 297, 334, 333, 332, 293, 298, 284]
    
    SPACED = [116, 111, 117, 118, 119, 100, 47, 126, 101, 123, 137, 177, 50, 36, 209, 129, 205, 147, 177, 187, 207, 206, 203, 10, 151, 9, 8, 8, 338, 337, 301, 251, 298, 333, 299, 297, 332, 284, 349, 348, 347, 346, 345, 447, 323, 280, 352, 330, 371, 358, 423, 426, 425, 427, 411, 376, 21, 71, 68, 54, 103, 104, 108, 69, 67, 109, 193, 417, 168, 188, 6, 412, 197, 174, 399, 456, 195, 236, 131, 51, 281, 360, 440, 4, 220, 219, 305, 227]
