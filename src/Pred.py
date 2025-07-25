#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :Pred.py
# @Time      :2025/7/25 23:05
# @Author    :ZHANG Yun in Rocket Force of University
# @Description: 
# @input     :
# @output    :
import torch
import pandas as pd
from preprocess import process_data
from torch.utils import data
import defaults
import torch.nn.functional as F
from model_bayesian_cnn import BayesianCNN
import numpy as np
import time
import Dropcnn
from scipy.stats import entropy
import obspy
import os


def pred(model, test_loader, num_samples=100, eval=True, uncertainty_aware=True):
    def entropy_along_axis(arr, axis=1):
        # 沿指定轴归一化得到概率分布
        prob = arr / np.sum(arr, axis, keepdims=True)
        entropies = np.apply_along_axis(lambda x: entropy(x, base=np.e), axis=1, arr=prob)
        return entropies

    if eval:
        model.eval()
    predictions = np.zeros((len(test_loader.dataset), num_samples, 2))
    Y_true = []
    t1 = time.time()
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            Y_true.append(target.item())
            data = torch.from_numpy(process_data(data.numpy(), sampling_rate=defaults.SAMPLING_RATE,
                                                 taper_fraction=defaults.TAPER_FRACTION,
                                                 highpass_cutoff=defaults.HIGH_PASS,
                                                 highpass_order=defaults.HIGH_PASS_ORDER,
                                                 lowpass_cutoff=defaults.LOW_PASS,
                                                 lowpass_order=defaults.LOW_PASS_ORDER)).to(device=defaults.DEVICE)
            for j in range(num_samples):
                output = model(data, sample=uncertainty_aware)  # 采样
                prob = F.softmax(output.cpu(), dim=1).squeeze().numpy()
                predictions[i, j] = prob
    pred_mean = np.mean(predictions, axis=1)
    entropy_values = entropy_along_axis(pred_mean, 1)
    pred_var = np.var(predictions, axis=1)
    pred_result = pd.DataFrame(
        {
            "True": Y_true,
            "mean_1": pred_mean[:, 0],
            "mean_2": pred_mean[:, 1],
            "entropy_values": entropy_values
        }
    )
    t2 = time.time()
    print(t2 - t1)
    time_using = pd.DataFrame(
        {
            "Time_Using(Seconds)": [t2 - t1]
        }
    )
    return pred_result, time_using
