#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :cnn.py
# @Time      :2025/4/8 23:22
# @Author    :ZHANG Yun in Rocket Force of University
# @Description: 
# @input     :
# @output    :

import torch.nn as nn
import torch.nn.functional as F

import defaults
import os
from defaults import CNN_TRAIN_RESULTS_SAVE, TIME_TRAIN_CNN_SAVE, MODEL_SAVE_CNN, TRAIN_LOSS_CNN_SAVE, \
    TRAIN_ACC_CNN_SAVE, TEST_ACC_SAVE_CNN


class CNN(nn.Module):
    """CNN"""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, stride=1, padding='same')
        self.pool = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding='same')

        self.conv3 = nn.Conv1d(128, 64, kernel_size=3, stride=1, padding='same')

        self.conv4 = nn.Conv1d(64, 16, kernel_size=3, stride=1, padding='same')

        # 全连接层
        self.fc1 = nn.Linear(4000, 16)
        self.fc2 = nn.Linear(16, 2)

        self.drop_out = nn.Dropout(defaults.DROPOUT)

        self.train_results_save_path = os.path.join(r"./results", "train_results_DropCNNp"+str(defaults.DROPOUT)+".csv")
        self.model_save_path = os.path.join(r"./model", "DropCNNp"+str(defaults.DROPOUT)+".pth")
        self.time_save_path = os.path.join(r"./results", "training_time_DropCNNp"+str(defaults.DROPOUT)+".csv")

    def forward(self, x, sample=False):
        """前向传播，可选择是否采样"""
        x = self.conv1(x)
        x = F.relu(x)
        x = self.drop_out(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.drop_out(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.drop_out(x)
        x = self.pool(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.drop_out(x)
        x = self.pool(x)

        x = x.view(x.shape[0], -1)

        x = F.relu(self.fc1(x))
        x = self.drop_out(x)
        x = self.fc2(x)
        return x

    def kl_loss(self, num_samples):
        """计算kl损失"""
        return 0.0

    def log_likelihood(self, x, y, num_samples=2):
        """计算 log_likelihood损失（变分推断目标）"""
        output = self(x, sample=False)
        log_likelihood = F.cross_entropy(output, y, reduction='sum')
        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(y.view_as(pred)).sum().item()
        return log_likelihood, correct
