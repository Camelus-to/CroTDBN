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


class TransPoolingEncoder(nn.Module):
    """
    Transformer encoder with Pooling mechanism.
    Input size: (batch_size, input_node_num, input_feature_size)
    Output size: (batch_size, output_node_num, input_feature_size)
    """

    def __init__(self, input_feature_size, input_node_num, hidden_size, output_node_num, pooling=True, orthogonal=True,
                 freeze_center=False, project_assignment=True):
        super().__init__()
        self.transformer = InterpretableTransformerEncoder(d_model=input_feature_size, nhead=4,
                                                           dim_feedforward=hidden_size,
                                                           batch_first=True)

        self.pooling = pooling
        if pooling:
            encoder_hidden_size = 32
            self.encoder = nn.Sequential(
                nn.Linear(input_feature_size *
                          input_node_num, encoder_hidden_size),
                nn.LeakyReLU(),
                nn.Linear(encoder_hidden_size, encoder_hidden_size),
                nn.LeakyReLU(),
                nn.Linear(encoder_hidden_size,
                          input_feature_size * input_node_num),
            )
            self.dec = DEC(cluster_number=output_node_num, hidden_dimension=input_feature_size, encoder=self.encoder,
                           orthogonal=orthogonal, freeze_center=freeze_center, project_assignment=project_assignment)

    def is_pooling_enabled(self):
        return self.pooling

    def forward(self, x):
        x = self.transformer(x)
        if self.pooling:
            x, assignment = self.dec(x)
            # loss = self.dec.loss(assignment)
            return x, assignment
        return x, None

    def get_attention_weights(self):
        return self.transformer.get_attention_weights()

    def loss(self, assignment):
        return self.dec.loss(assignment)

    def compute_cluster_purities(assignments, true_labels):
        """
        计算每个簇的纯度。
        :param assignments: [batch_size * node_num, cluster_number] 聚类分配结果
        :param true_labels: [batch_size * node_num] 真实的标签
        :return: [cluster_number] 每个簇的纯度
        """
        cluster_number = assignments.size(-1)
        cluster_purities = torch.zeros(cluster_number).to(assignments.device)
        cluster_labels = torch.argmax(assignments, dim=1)  # [batch_size * node_num]

        for k in range(cluster_number):
            indices = (cluster_labels == k)
            if indices.sum() == 0:
                continue
            labels_in_cluster = true_labels[indices]
            most_common_label = labels_in_cluster.mode()[0]
            purity = (labels_in_cluster == most_common_label).float().mean()
            cluster_purities[k] = purity
        return cluster_purities


class Transformer_duibi(BaseModel):

    def __init__(self, config: DictConfig, semantic_similarity_matrix):

        super().__init__()
        self.node_feature_dim_dynamic = 116
        self.attention_list = nn.ModuleList()
        forward_dim = config.dataset.node_sz

        self.pos_encoding = config.model.pos_encoding
        self.prompt_tokens = semantic_similarity_matrix
        if self.pos_encoding == 'identity':
            if self.prompt_tokens is None:
                # 如果 prompt_tokens 不存在，初始化 node_identity 作为位置编码
                self.node_identity = nn.Parameter(torch.zeros(
                    config.dataset.node_sz, config.model.pos_embed_dim), requires_grad=True)
                forward_dim = config.dataset.node_sz + config.model.pos_embed_dim
                nn.init.kaiming_normal_(self.node_identity)
            else:
                # 如果 prompt_tokens 存在，forward_dim 要加上 prompt_tokens 的维度
                # forward_dim = config.dataset.node_sz + self.prompt_tokens.shape[-1]
                forward_dim = config.dataset.node_sz

        sizes = config.model.sizes
        sizes[0] = config.dataset.node_sz
        in_sizes = [config.dataset.node_sz] + sizes[:-1]
        do_pooling = config.model.pooling
        self.do_pooling = do_pooling
        self.static_trans_pooling = TransPoolingEncoder(
            input_feature_size=116,  # 簇的特征大小
            input_node_num=116,  # 将6个时间段的100个簇合并
            hidden_size=1024,
            output_node_num=100,  # 再次聚类为100个簇
            pooling=False,
            orthogonal=config.model.orthogonal,
            freeze_center=config.model.freeze_center,
            project_assignment=config.model.project_assignment)

        self.static_trans_pooling2 = TransPoolingEncoder(
            input_feature_size=116,  # 簇的特征大小
            input_node_num=116,  # 将6个时间段的100个簇合并
            hidden_size=1024,
            output_node_num=100,  # 再次聚类为100个簇
            pooling=True,
            orthogonal=config.model.orthogonal,
            freeze_center=config.model.freeze_center,
            project_assignment=config.model.project_assignment)

        self.dim_reduction = nn.Sequential(
            nn.Linear(forward_dim, 8),
            nn.LeakyReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(800, 256),
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

    def forward(self,
                time_seires: torch.tensor,
                node_feature: torch.tensor,
                dynamic_fc: torch.tensor,
                semantic_similarity_matrix: torch.tensor):

        bz, _, _, = node_feature.shape  # 该模型的输入 node_feature是200*200的，就是一个功能连接矩阵，而 time_seires是200*100的，是时间序列
        bz, time_steps, num_nodes, _ = dynamic_fc.shape

        assignments = []
        # attention_list 是 2个ModuleList

        node_feature_static, assignment_static = self.static_trans_pooling(node_feature)  # 不带聚类的
        node_feature_static, assignment_static = self.static_trans_pooling2(node_feature)  # 不带聚类的
        node_feature_static = self.dim_reduction(
            node_feature_static)  # 进行了维度的缩减，原来是16*100*116，经过缩减维度以后最后一个特征的维度变成了8，整体的维度变成了16*100*8；
        node_feature_static = node_feature_static.reshape((bz, -1))  # 变成了16*800的
        # 最终结合静态特征和动态聚类后的特征

        combined_feature = torch.cat([node_feature_static], dim=-1)

        outputs = self.fc(combined_feature)

        # print(contra_loss)
        contra_loss = None
        return outputs, contra_loss

    def adj_to_edge_index(self, adj_matrix):
        # 将邻接矩阵转换为 edge_index 和 edge_weight
        edge_index = (adj_matrix > 0).nonzero(as_tuple=False).t().contiguous()
        edge_weight = adj_matrix[adj_matrix > 0]
        return edge_index, edge_weight

    def get_attention_weights(self):
        return [atten.get_attention_weights() for atten in self.attention_list]

    def get_cluster_centers(self) -> torch.Tensor:
        """
        Get the cluster centers, as computed by the encoder.

        :return: [number of clusters, hidden dimension] Tensor of dtype float
        """
        return self.dec.get_cluster_centers()

    def loss(self, assignments):
        """
        Compute KL loss for the given assignments. Note that not all encoders contain a pooling layer.
        Inputs: assignments: [batch size, number of clusters]
        Output: KL loss
        """
        decs = list(
            filter(lambda x: x.is_pooling_enabled(), self.attention_list))
        assignments = list(filter(lambda x: x is not None, assignments))
        loss_all = None

        for index, assignment in enumerate(assignments):
            if loss_all is None:
                loss_all = decs[index].loss(assignment)
            else:
                loss_all += decs[index].loss(assignment)
        return loss_all
