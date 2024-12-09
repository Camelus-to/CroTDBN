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


class PositionalEncodingMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(PositionalEncodingMLP, self).__init__()
        # 定义 MLP 层，用于处理位置编码
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, pos_emb):
        # 输入是位置编码，输出的维度与输入一致
        return self.mlp(pos_emb)


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
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4):
        super(GATModel, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight)
        x = self.relu(x)
        x = self.conv2(x, edge_index, edge_weight)
        return x


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


class BrainNetworkTransformer_test(BaseModel):

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
        for index, size in enumerate(sizes):
            self.attention_list.append(
                TransPoolingEncoder(input_feature_size=forward_dim,
                                    input_node_num=in_sizes[index],
                                    hidden_size=1024,
                                    output_node_num=100,
                                    pooling=do_pooling[index],
                                    orthogonal=config.model.orthogonal,
                                    freeze_center=config.model.freeze_center,
                                    project_assignment=config.model.project_assignment))

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
        # 时间维度的 Transformer
        self.temporal_transformer = TimeAttentionTransformer(116, 116, 116, nhead=4, num_layers=4)

        # 定义时间注意力机制
        self.time_attention = nn.Sequential(
            nn.Linear(self.node_feature_dim_dynamic, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

        # 对6个时间段的簇再次进行最终的聚类
        self.final_trans_pooling = TransPoolingEncoder(
            input_feature_size=116,  # 簇的特征大小
            input_node_num=116 * 6,  # 将6个时间段的100个簇合并
            hidden_size=1024,
            output_node_num=100,  # 再次聚类为100个簇
            pooling=True,
            orthogonal=config.model.orthogonal,
            freeze_center=config.model.freeze_center,
            project_assignment=config.model.project_assignment)
        self.pos_encoding_mlp = PositionalEncodingMLP(input_dim=116)

    def forward(self,
                time_seires: torch.tensor,
                node_feature: torch.tensor,
                dynamic_fc: torch.tensor,
                semantic_similarity_matrix: torch.tensor):

        bz, _, _, = node_feature.shape  # 该模型的输入 node_feature是200*200的，就是一个功能连接矩阵，而 time_seires是200*100的，是时间序列
        bz, time_steps, num_nodes, _ = dynamic_fc.shape
        self.prompt_tokens = self.pos_encoding_mlp(self.prompt_tokens)

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
        # 将所有时间步的输出拼接为 [batch_size, time_steps, num_nodes, feature_dim]
        temporal_input = torch.stack(all_outputs, dim=1)  # 把时间维度也拼起来

        if self.pos_encoding == 'identity':
            if self.prompt_tokens is not None:
                pos_emb = self.prompt_tokens.unsqueeze(0).unsqueeze(0).expand(bz, time_steps, *self.prompt_tokens.shape)
                temporal_output = temporal_input + pos_emb
            else:
                pos_emb = self.node_identity.expand(bz, *self.node_identity.shape)
                node_feature = torch.cat([node_feature, pos_emb], dim=-1)

        temporal_output = self.temporal_transformer(temporal_output)
        # 进行聚类操作
        final_repr = temporal_output.reshape(bz, -1, 116)  # [batch_size, num_nodes, feature_dim]
        node_feature_dynamic, assignment_dynamic = self.final_trans_pooling(final_repr)
        node_feature_dynamic = self.dim_reduction_final(node_feature_dynamic)
        node_feature_dynamic = node_feature_dynamic.reshape((bz, -1))

        assignments = []

        if self.pos_encoding == 'identity':
            if self.prompt_tokens is not None:
                pos_emb = self.prompt_tokens.unsqueeze(0).expand(bz, *self.prompt_tokens.shape)
                node_feature = node_feature + pos_emb
            else:
                pos_emb = self.node_identity.expand(bz, *self.node_identity.shape)
                node_feature = torch.cat([node_feature, pos_emb], dim=-1)

        # attention_list 是 2个ModuleList
        for atten in self.attention_list:
            node_feature_static, assignment_static = atten(node_feature)
            assignments.append(assignment_static)

        node_feature_static = self.dim_reduction(
            node_feature_static)  # 进行了维度的缩减，原来是16*100*116，经过缩减维度以后最后一个特征的维度变成了8，整体的维度变成了16*100*8；
        node_feature_static = node_feature_static.reshape((bz, -1))  # 变成了16*800的
        # 最终结合静态特征和动态聚类后的特征

        combined_feature = torch.cat([node_feature_static, node_feature_dynamic], dim=-1)

        outputs = self.fc(combined_feature)

        # print(total_loss)

        return outputs

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
