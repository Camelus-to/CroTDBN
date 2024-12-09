import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer
from .ptdec import DEC
from typing import List
from .components import InterpretableTransformerEncoder
from torch_geometric.nn import GATConv
from torch_geometric.data import Data, Batch
from omegaconf import DictConfig
from omegaconf import DictConfig
from ..base import BaseModel


class TimeAttention(nn.Module):
    def __init__(self, input_dim):
        super(TimeAttention, self).__init__()
        self.attention_layer = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        """
        输入x的形状为 [batch_size, time_steps, num_nodes, feature_dim]
        返回经过时间注意力加权后的特征
        """
        batch_size, time_steps, num_nodes, feature_dim = x.shape
        # 计算注意力权重 [batch_size, time_steps, num_nodes, 1]
        attn_weights = self.attention_layer(x).squeeze(-1)
        attn_weights = torch.softmax(attn_weights, dim=1)  # 对时间维度softmax

        # 使用注意力权重对特征加权 [batch_size, time_steps, num_nodes, feature_dim]
        weighted_x = x * attn_weights.unsqueeze(-1)

        # 将加权后的特征沿时间维度求和，得到 [batch_size, num_nodes, feature_dim]
        output = torch.sum(weighted_x, dim=1)
        return output, attn_weights


import torch.nn.functional as F


# 计算特征表示的相似度
def compute_similarity(features1, features2):
    features1 = F.normalize(features1, dim=-1)
    features2 = F.normalize(features2, dim=-1)
    return torch.matmul(features1, features2.transpose(-1, -2))  # [batch_size, num_nodes, num_nodes]


# 生成对比学习的损失
def contrastive_loss(similarity_matrix):
    batch_size, num_nodes, _ = similarity_matrix.shape
    labels = torch.arange(num_nodes).unsqueeze(0).expand(batch_size, -1).to(similarity_matrix.device)
    loss = F.cross_entropy(similarity_matrix, labels)
    return loss


class TimeAttentionTransformer(nn.Module):
    def __init__(self, num_nodes, feature_dim, d_model, nhead, num_layers, dim_feedforward=512, dropout=0.1):
        super(TimeAttentionTransformer, self).__init__()

        # 线性层将 feature_dim 压缩到 d_model
        self.input_projection = nn.Linear(feature_dim, d_model)

        # 定义 Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,  # 输入的嵌入维度
            nhead=nhead,  # 注意力头的数量
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 线性层将 Transformer 的输出映射回 feature_dim
        self.output_projection = nn.Linear(d_model, feature_dim)

    def forward(self, x):
        """
        输入 x 的形状: [batch_size, time_steps, num_nodes, feature_dim]
        """

        batch_size, time_steps, num_nodes, feature_dim = x.shape

        # Step 1: 将特征维度投影到 d_model
        x = self.input_projection(x)  # [batch_size, time_steps, num_nodes, d_model]

        # Step 2: 将输入调整为 [batch_size * num_nodes, time_steps, d_model]
        # 这样，每个节点独立处理其时间序列
        x = x.permute(2, 0, 1, 3).contiguous()  # [num_nodes, batch_size, time_steps, d_model]
        x = x.view(batch_size * num_nodes, time_steps, -1)  # [batch_size * num_nodes, time_steps, d_model]

        # Step 3: 传递到 Transformer 中，处理时间步之间的依赖关系
        x = self.transformer_encoder(x)  # [batch_size * num_nodes, time_steps, d_model]

        # Step 4: 恢复回 [num_nodes, batch_size, time_steps, d_model]
        x = x.view(num_nodes, batch_size, time_steps, -1).permute(1, 2, 0,
                                                                  3)  # [batch_size, time_steps, num_nodes, d_model]

        # Step 5: 输出投影回原始的 feature_dim
        x = self.output_projection(x)  # [batch_size, time_steps, num_nodes, feature_dim]

        return x


class GATModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, threshold_percent=0.1):
        super(GATModel, self).__init__()
        self.threshold_percent = threshold_percent  # 阈值百分比
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, edge_weight):
        num_edges = edge_weight.size(0)
        k = int(num_edges * self.threshold_percent)  # 保留前10%的边

        # Step 2: 获取边权重的阈值
        threshold_value = torch.topk(edge_weight, k, largest=True).values.min()

        # Step 3: 根据阈值筛选边
        mask = edge_weight >= threshold_value
        filtered_edge_index = edge_index[:, mask]  # 只保留权重高于阈值的边
        filtered_edge_weight = edge_weight[mask]  # 保留对应的边权重

        x = self.conv1(x, filtered_edge_index, filtered_edge_weight)
        x = self.relu(x)
        x = self.conv2(x, filtered_edge_index, filtered_edge_weight)
        return x


class GAT(BaseModel):

    def __init__(self, config: DictConfig, semantic_similarity_matrix):

        super().__init__()
        self.node_feature_dim_dynamic = 116
        self.attention_list = nn.ModuleList()
        forward_dim = config.dataset.node_sz
        self.prompt_tokens = semantic_similarity_matrix
        sizes = config.model.sizes
        sizes[0] = config.dataset.node_sz
        in_sizes = [config.dataset.node_sz] + sizes[:-1]
        do_pooling = config.model.pooling
        self.do_pooling = do_pooling

        self.dim_reduction = nn.Sequential(
            nn.Linear(forward_dim, 8),
            nn.LeakyReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(1600, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 2)
        )

        self.dim_reduction_final = nn.Sequential(
            nn.Linear(self.node_feature_dim_dynamic, 8),
            nn.LeakyReLU()
        )

        self.num_nodes = 116  # 节点数量，例如116
        self.node_feature_dim = 116  # 节点特征维度，例如3
        # 时间注意力机制：用于对dynamic_fc处理
        # 定义空间和时间注意力机制
        # 初始化 GAT 模型
        self.gat = GATModel(
            in_channels=self.node_feature_dim_dynamic,
            hidden_channels=int(self.node_feature_dim_dynamic // 2),
            out_channels=self.node_feature_dim_dynamic,
            heads=4
        )
        # 最终分类层
        self.classifier = nn.Sequential(
            nn.Flatten(),  # 展平所有节点的特征
            nn.Linear(self.node_feature_dim_dynamic * self.num_nodes, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # 假设是二分类
        )

    def forward(self,
                time_seires: torch.tensor,
                node_feature: torch.tensor,
                dynamic_fc: torch.tensor,
                semantic_similarity_matrix: torch.tensor):

        bz, _, _, = node_feature.shape  # 该模型的输入 node_feature是200*200的，就是一个功能连接矩阵，而 time_seires是200*100的，是时间序列
        bz, time_steps, num_nodes, _ = dynamic_fc.shape
        all_outputs = []
        for t in range(time_steps):
            # 获取当前时间步的功能连接矩阵
            adj_matrix = dynamic_fc[:, t, :, :]  # [batch_size, num_nodes, num_nodes]
            for i in range(adj_matrix.size(0)):
                adj_matrix[i, i] = 0
            # 将功能连接矩阵转换为图数据
            graphs = []
            for i in range(bz):
                edge_index, edge_weight = self.adj_to_edge_index(adj_matrix[i])
                x = node_feature[i]  # [num_nodes, node_feature_dim]
                data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight)
                graphs.append(data)
            batch = Batch.from_data_list(graphs)

            # 通过 GAT 模型
            gat_output = self.gat(batch.x, batch.edge_index, batch.edge_weight)
            # 将输出重新分割为 [batch_size, num_nodes, feature_dim]
            gat_output = gat_output.view(bz, num_nodes, -1)
            all_outputs.append(gat_output)
            # 将时间步的所有输出在时间维度上求平均
        combined_output = torch.stack(all_outputs, dim=1).mean(dim=1)  # [batch_size, num_nodes, feature_dim]
        # 最终分类层
        outputs = self.classifier(combined_output)  # [batch_size, 2]
        # 假设有一个对比学习的损失
        contra_loss = None  # 计算对比学习损失的部分可以根据需要填充
        return outputs, contra_loss

    def adj_to_edge_index(self, adj_matrix):
        # 将邻接矩阵转换为 edge_index 和 edge_weight
        edge_index = (adj_matrix > 0).nonzero(as_tuple=False).t().contiguous()
        edge_weight = adj_matrix[adj_matrix > 0]
        return edge_index, edge_weight

    def get_attention_weights(self):
        return [atten.get_attention_weights() for atten in self.attention_list]
