# -*- coding: utf-8 -*-
"""
动作识别模型定义
1D-CNN + LSTM 轻量级时序模型
"""
import torch
import torch.nn as nn
from config import *


class ActionRecognitionModel(nn.Module):
    """
    1D-CNN + LSTM 动作识别模型

    结构：
    输入 (batch, 30, 57) ->
    Conv1D(57->64, k=3) + ReLU + MaxPool ->
    Conv1D(64->128, k=3) + ReLU + MaxPool ->
    LSTM(128->64) ->
    FC(64->64) + ReLU + Dropout(0.3) ->
    FC(64->4) + Softmax

    参数量约 8.9 万
    """

    def __init__(self, input_dim=57, num_classes=NUM_CLASSES,
                 cnn_ch1=CNN_CHANNELS_1, cnn_ch2=CNN_CHANNELS_2,
                 kernel_size=CNN_KERNEL_SIZE, lstm_hidden=LSTM_HIDDEN_SIZE,
                 dropout=DROPOUT_RATE):
        super(ActionRecognitionModel, self).__init__()

        # 1D卷积层 - 提取短时局部动态特征
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_dim, cnn_ch1, kernel_size=kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(cnn_ch1, cnn_ch2, kernel_size=kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        # LSTM层 - 整合长时间范围上下文
        self.lstm = nn.LSTM(
            input_size=cnn_ch2,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )

        # 全连接分类层
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, num_classes)
        )

    def forward(self, x):
        """
        x: (batch, seq_len, feature_dim) = (batch, 30, 57)
        """
        # Conv1D expects (batch, channels, seq_len)
        x = x.permute(0, 2, 1)  # (batch, 57, 30)

        x = self.conv1(x)  # (batch, 64, 15)
        x = self.conv2(x)  # (batch, 128, 7)

        # 转回 (batch, seq_len, features) 给 LSTM
        x = x.permute(0, 2, 1)  # (batch, 7, 128)

        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out: (batch, 7, 64)

        # 取最后一个时步的隐藏状态
        x = lstm_out[:, -1, :]  # (batch, 64)

        # 分类
        x = self.classifier(x)  # (batch, 4)

        return x


class PureLSTMModel(nn.Module):
    """
    纯 LSTM 模型（用于对比实验）
    """

    def __init__(self, input_dim=57, num_classes=NUM_CLASSES,
                 lstm_hidden=LSTM_HIDDEN_SIZE, dropout=DROPOUT_RATE):
        super(PureLSTMModel, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=False,
            dropout=0.2
        )

        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, num_classes)
        )

    def forward(self, x):
        """
        x: (batch, seq_len, feature_dim) = (batch, 30, 57)
        """
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]
        x = self.classifier(x)
        return x


def count_parameters(model):
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 测试模型结构
    model = ActionRecognitionModel()
    print("1D-CNN + LSTM 模型结构:")
    print(model)
    print(f"\n参数量: {count_parameters(model):,}")

    # 测试前向传播
    dummy_input = torch.randn(4, WINDOW_SIZE, 57)
    output = model(dummy_input)
    print(f"\n输入形状: {dummy_input.shape}")
    print(f"输出形状: {output.shape}")

    # 对比模型
    lstm_model = PureLSTMModel()
    print(f"\n纯 LSTM 模型参数量: {count_parameters(lstm_model):,}")
