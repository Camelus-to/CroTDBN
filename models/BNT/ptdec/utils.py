import torch
from torch_geometric.data import Data


def create_graph_data(node_feature: torch.Tensor, assignments: torch.Tensor):
    """
    生成图数据，节点特征由原始特征和聚类分配结果拼接，边根据聚类分配与注意力机制构造。

    :param node_feature: 原始输入节点特征 [batch_size, num_nodes, feature_dim]
    :param assignments: 聚类分配矩阵 [batch_size, num_nodes, num_clusters]
    :return: 适应下游任务的图数据 (x, edge_index, batch, edge_attr)
    """

    def construct_node_features(node_feature: torch.Tensor, assignments: torch.Tensor):
        # 将原始特征和聚类分配结果拼接
        node_features = torch.cat([assignments], dim=-1)
        return node_features

    def construct_weighted_graph(node_feature: torch.Tensor):

        adjacency_matrices = node_feature

        return adjacency_matrices

    node_features = construct_node_features(node_feature, assignments)

    adjacency_matrices = construct_weighted_graph(node_feature)

    batch_size, num_nodes, _ = node_features.shape
    x_list = []
    edge_index_list = []
    edge_attr_list = []
    batch_list = []

    for i in range(batch_size):
        edge_index = []
        edge_weight = []

        for j in range(num_nodes):
            for k in range(num_nodes):
                if adjacency_matrices[i, j, k] > 0:  # 保留有权重的边
                    edge_index.append([j, k])
                    edge_weight.append(adjacency_matrices[i, j, k])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # [2, num_edges]
        edge_weight = torch.tensor(edge_weight, dtype=torch.float)  # [num_edges]

        x = node_features[i]  # [num_nodes, feature_dim + num_clusters]

        batch = torch.full((num_nodes,), i, dtype=torch.long)

        x_list.append(x)
        edge_index_list.append(edge_index)
        edge_attr_list.append(edge_weight)
        batch_list.append(batch)

    x = torch.cat(x_list, dim=0)  # [total_num_nodes, feature_dim + num_clusters]
    edge_index = torch.cat(edge_index_list, dim=1)  # [2, total_num_edges]
    edge_attr = torch.cat(edge_attr_list, dim=0)  # [total_num_edges]
    batch = torch.cat(batch_list, dim=0)  # [total_num_nodes]

    return x, edge_index, batch, edge_attr
