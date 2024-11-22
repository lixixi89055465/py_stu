# -*- coding: utf-8 -*-
# @Time : 2024/10/3 8:45
# @Author : nanji
# @Site : https://www.bilibili.com/video/BV1kJWUeoEuG/?p=4&spm_id_from=pageDriver&vd_source=50305204d8a1be81f31d861b12d4d5cf
# @File : InstructionTuningLab.py
# @Software: PyCharm 
# @Comment :
import itertools
import jsonlines
from datasets import load_dataset
from pprint import pprint

from llama import BasicModelRunner
