import torch
import torch.nn as nn
import torch.optim as optim


class FlattenClassifier(nn.Module):
    def __init__(self, input_dim=116 * 116, hidden_dim=128, output_dim=2):
        super(FlattenClassifier, self).__init__()

        # 定义一个简单的前馈神经网络
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # 输入展平后的向量
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)  # 输出为二分类
        )

    def forward(self, x):
        # 将输入矩阵展平为一维向量
        x = x.view(x.size(0), -1)  # 这里假设输入形状是 (batch_size, 116, 116)
        return self.model(x)
