#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :run.py
# @Time      :2025/7/25 23:12
# @Author    :ZHANG Yun in Rocket Force of University
# @Description: 
# @input     :
# @output    :
from torch.utils import data
import numpy as np
import obspy

import Dropcnn
import defaults
import os
import pandas as pd
from model_bayesian_cnn import BayesianCNN
import torch
from Pred import pred


class Dataset(data.Dataset):
    def __init__(self, csv):
        super().__init__()
        self.csv = csv

    def __getitem__(self, id):
        data = np.expand_dims(obspy.read(self.csv.fpath.iloc[id])[0].data.T, 0)
        label = defaults.MAPPING.get(self.csv.lable.iloc[id])
        return data, label

    def __len__(self):
        return len(self.csv)


def build_test_loader(csv):
    test = Dataset(csv)
    test_dataloader = data.DataLoader(test, batch_size=1, shuffle=False, num_workers=0)
    return test_dataloader


if __name__ == "__main__":
    # Load test data
    csv = pd.read_csv(r"./test.csv")
    # Build test dataloader
    test_dataloader = build_test_loader(csv)
    # Load Bayesian CNN model
    model = BayesianCNN().to(defaults.DEVICE)
    model.load_state_dict(torch.load(r"../model/BAYESIAN_CNN.pth"))
    # Set model to evaluation mode
    """
        uncertainty_aware: True: Use Bayesian CNN sampling with uncertainty-aware mode,
                           False: Use Bayesian CNN predictions with uncertainty-free mode.
        
        num_samples: Number of prediction for uncertainty estimation.
        eval: Set to True to evaluate the model.
        model: The model to use for predictions.
    """
    pred_result, timeused = pred(model, test_dataloader, num_samples=10, eval=True, uncertainty_aware=True)

    model = Dropcnn.CNN().to(defaults.DEVICE)
    model.load_state_dict(torch.load(r"../model/DropCNNp0.1.pth"))
    """
        uncertainty_aware: True: Use DropCNN predictions with uncertainty-aware mode,
                           False: Use DropCNN predictions with uncertainty-free mode.
        eval: False to use the model in uncertainty-aware mode.
              True to use the model in uncertainty-free mode.
        num_samples: Number of prediction for uncertainty estimation.
    """
    pred_result, timeused = pred(model, test_dataloader, num_samples=10, eval=False, uncertainty_aware=True)

