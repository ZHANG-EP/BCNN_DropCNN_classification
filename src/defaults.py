#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :defaults.py
# @Time      :2025/7/25 23:09
# @Author    :ZHANG Yun in Rocket Force of University
# @Description: 
# @input     :
# @output    :
import torch

CLASS = ['eq', 'ep']
MAPPING = {CLASS[0]: 0, CLASS[1]: 1, "ss": 2, "sp": 3}
SAMPLING_RATE = 50
TAPER_FRACTION = 0.01
HIGH_PASS = 1
HIGH_PASS_ORDER = 4
LOW_PASS = 20
LOW_PASS_ORDER = 4
DROPOUT = 0.1
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
