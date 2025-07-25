#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :model_bayesian_cnn.py
# @Time      :2025/4/2 19:05
# @Author    :ZHANG Yun in Rocket Force of University
# @Description: 
# @input     :
# @output    :
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal
from defaults import BAYESIAN_CNN_TRAIN_RESULTS_SAVE, MODEL_SAVE_PATH_BAYESIAN_CNN, TIME_TRAIN_BAYESIAN_CNN_SAVE, \
    TRAIN_LOSS_BAYESIAN_CNN_SAVE, TRAIN_ACC_BAYESIAN_CNN_SAVE

# 设置随机种子以保证可重复性
torch.manual_seed(42)
np.random.seed(42)

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 定义贝叶斯层
class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # 权重分布参数
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        # self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))

        # 偏置分布参数
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        # self.bias_rho = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))

        # 初始化
        self.init_parameters()

    def init_parameters(self):
        # 初始化参数
        nn.init.kaiming_uniform_(self.weight_mu, mode='fan_in')
        nn.init.uniform_(self.bias_mu, -0.2, 0.2)
        nn.init.constant_(self.weight_sigma, -5)
        nn.init.constant_(self.bias_sigma, -5)

    def forward(self, x, sample=True):
        # 计算sigma (log(1 + exp(rho)))
        # self.weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        # self.bias_sigma = torch.log1p(torch.exp(self.bias_rho))
        if sample:
            weight = Normal(self.weight_mu, F.softplus(self.weight_sigma)).rsample()
            bias = Normal(self.bias_mu, F.softplus(self.bias_sigma)).rsample()
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        # 计算输出
        return F.linear(x, weight, bias)

    def kl_divergence(self):
        # 计算KL散度 (变分分布与先验分布之间的KL散度)
        kl_divergence = 0
        q_weight = Normal(self.weight_mu, F.softplus(self.weight_sigma))
        q_bias = Normal(self.bias_mu, F.softplus(self.bias_sigma))

        # 先验（标准正态）
        p_weight = Normal(0, 1)
        p_bias = Normal(0, 1)

        # 计算 KL 散度
        kl_divergence += torch.distributions.kl_divergence(q_weight, p_weight).mean()
        kl_divergence += torch.distributions.kl_divergence(q_bias, p_bias).mean()
        return kl_divergence


class BayesianConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding="same"):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # 权重分布参数
        self.weight_mu = nn.Parameter(
            torch.Tensor(out_channels, in_channels, kernel_size))
        # self.weight_rho = nn.Parameter(
        #     torch.Tensor(out_channels, in_channels, kernel_size))
        self.weight_sigma = nn.Parameter(
            torch.Tensor(out_channels, in_channels, kernel_size))

        # 偏置分布参数
        self.bias_mu = nn.Parameter(torch.Tensor(out_channels))
        # self.bias_rho = nn.Parameter(torch.Tensor(out_channels))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_channels))
        # 初始化
        self.init_parameters()

    def init_parameters(self):
        nn.init.kaiming_uniform_(self.weight_mu, mode='fan_in')
        nn.init.uniform_(self.bias_mu, -0.2, 0.2)
        nn.init.constant_(self.weight_sigma, -5)
        nn.init.constant_(self.bias_sigma, -5)

    def forward(self, x, sample=True):
        # 计算sigma (log(1 + exp(rho)))
        # self.weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        # self.bias_sigma = torch.log1p(torch.exp(self.bias_rho))
        if sample:
            weight = Normal(self.weight_mu, F.softplus(self.weight_sigma)).rsample()
            bias = Normal(self.bias_mu, F.softplus(self.bias_sigma)).rsample()
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        # 计算卷积输出
        return F.conv1d(x, weight, bias, stride=self.stride, padding=self.padding)

    def kl_divergence(self):
        # 计算KL散度
        kl_divergence = 0
        q_weight = Normal(self.weight_mu, F.softplus(self.weight_sigma))
        q_bias = Normal(self.bias_mu, F.softplus(self.bias_sigma))

        # 先验（标准正态）
        p_weight = Normal(0, 1)
        p_bias = Normal(0, 1)

        # 计算 KL 散度
        kl_divergence += torch.distributions.kl_divergence(q_weight, p_weight).mean()
        kl_divergence += torch.distributions.kl_divergence(q_bias, p_bias).mean()
        return kl_divergence


# 定义贝叶斯CNN模型
class BayesianCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 卷积层
        self.conv1 = BayesianConv1d(1, 64, kernel_size=3, stride=1, padding='same')
        self.pool = nn.MaxPool1d(kernel_size=2)

        self.conv2 = BayesianConv1d(64, 128, kernel_size=3, stride=1, padding='same')

        self.conv3 = BayesianConv1d(128, 64, kernel_size=3, stride=1, padding='same')

        self.conv4 = BayesianConv1d(64, 16, kernel_size=3, stride=1, padding='same')

        # 全连接层
        self.fc1 = BayesianLinear(4000, 16)
        self.fc2 = BayesianLinear(16, 2)

        self.train_results_save_path = BAYESIAN_CNN_TRAIN_RESULTS_SAVE
        self.model_save_path = MODEL_SAVE_PATH_BAYESIAN_CNN
        self.time_save_path = TIME_TRAIN_BAYESIAN_CNN_SAVE

        self.train_loss_save = TRAIN_LOSS_BAYESIAN_CNN_SAVE
        self.train_acc_save = TRAIN_ACC_BAYESIAN_CNN_SAVE

    def forward(self, x, sample=False):

        x = self.conv1(x, sample)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv2(x, sample)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv3(x, sample)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv4(x, sample)
        x = F.relu(x)
        x = self.pool(x)

        x = x.view(x.shape[0], -1)

        x = F.relu(self.fc1(x, sample))
        x = self.fc2(x, sample)

        return x

    def kl_loss(self, num_samples=1):
        kl_divergence = 0
        # 蒙特卡洛采样

        # 计算所有层的KL散度之和
        for module in self.modules():
            if isinstance(module, (BayesianLinear, BayesianConv1d)):
                kl_divergence += module.kl_divergence()
        return kl_divergence

    def log_likelihood(self, x, y, num_samples):
        """计算 log_likelihood损失（变分推断目标）"""
        log_likelihood = 0
        # 蒙特卡洛采样
        correct = 0
        for _ in range(num_samples):
            output = self(x, sample=True)
            log_likelihood += F.cross_entropy(output, y, reduction='sum')
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(y.view_as(pred)).sum().item()

        correct /= num_samples
        # 平均采样损失
        log_likelihood /= num_samples
        return log_likelihood, correct
