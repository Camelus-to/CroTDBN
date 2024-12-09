import torch
import torch.nn as nn
import torch.optim as optim


class TransformerModel(nn.Module):
    def __init__(self, num_brain_regions, hidden_dim, num_classes):
        super(TransformerModel, self).__init__()
        # Define the encoder layer and transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.fc = nn.Linear(num_brain_regions * hidden_dim, num_classes)

    def forward(self, x):
        # Input x: [batch_size, num_brain_regions, hidden_dim]
        x = self.transformer(x)  # Apply transformer encoder to input
        x = x.view(x.size(0), -1)  # Flatten the output
        output = self.fc(x)  # Fully connected layer for classification
        return output


# Ising-like energy function
def energy_function(S, J, h):
    """
    S: Signal states for each brain region (tensor)  最终记录的每个脑区的状态，也就是每个脑区的值 S是16*10的，16个样本 10个脑区
    经过 torch.matmul(S.unsqueeze(1), S.unsqueeze(2))) 变成了16*1*1的
    J: Interaction strength between regions (matrix) 功能连接矩阵 是10*10的
    h: External field for each region (vector)
    """
    s1 = S.unsqueeze(2)
    s2 = S.unsqueeze(1)
    s = s1 @ s2  # 使用广播机制进行外积
    interaction_term = -torch.sum(J * s)  # S_i * J_ij * S_j
    field_term = -torch.sum(h * S)  # h_i * S_i
    return interaction_term + field_term


# Example classification and energy loss combination
def combined_loss_fn(logits, labels, S, J, h, classification_loss_fn, alpha=0.1):
    """
    logits: Output from the Transformer
    labels: Ground truth labels
    S: Signal states for each brain region
    J: Interaction matrix
    h: External field
    classification_loss_fn: Classification loss function (e.g., CrossEntropy)
    alpha: Weight for the energy loss term
    """
    # Classification loss
    classification_loss = classification_loss_fn(logits, labels)

    # Energy loss (minimization)
    energy_loss = energy_function(S, J, h)

    # Combined loss
    total_loss = classification_loss + alpha * energy_loss
    return total_loss

import torch

S = torch.randn(16, 10)  # 假设 S 是 16x10 的随机张量
s1 = S.unsqueeze(2)  # 将维度变为 16x1x10
s2 = S.unsqueeze(1)  # 将维度变为 16x10x1
s = s1 @ s2  # 使用广播机制进行外积  # 矩阵乘法，结果应为 16x10x10

print(s.shape)  # 输出应为 torch.Size([16, 10, 10])
# Example usage
num_brain_regions = 10  # Assume 10 brain regions
hidden_dim = 64
num_classes = 2
model = TransformerModel(num_brain_regions, hidden_dim, num_classes)

# Sample inputs
batch_size = 16
x = torch.rand(batch_size, num_brain_regions, hidden_dim)  # Random input signals
labels = torch.randint(0, num_classes, (batch_size,))  # Random labels

# Signal states S, interaction matrix J J是功能连接矩阵, and external field h
S = torch.rand(batch_size, num_brain_regions)  # Signal states for each brain region
J = torch.rand(num_brain_regions, num_brain_regions)  # Interaction matrix (e.g., connectivity)
h = torch.rand(num_brain_regions)  # External field

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Forward pass
logits = model(x)

# Combined loss
loss = combined_loss_fn(logits, labels, S, J, h, criterion, alpha=0.1)

# Backward and optimize
optimizer.zero_grad()
loss.backward()
optimizer.step()

print(f"Loss: {loss.item()}")
