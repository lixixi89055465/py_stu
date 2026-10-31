import base64
import copy
import os
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from onnxocr import get_logger

log = get_logger("license_plate")

PLATE_CHARS = (
    "#京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新"
    "学警港澳挂使领民航危0123456789ABCDEFGHJKLMNPQRSTUVWXYZ险品"
)
MEAN_VALUE = 0.588
STD_VALUE = 0.193

print('aaaaaaaa')