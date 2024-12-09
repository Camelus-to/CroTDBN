import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel  # 可以使用RBF核计算相似度矩阵
from torch.nn import Parameter
import torch.nn as nn
from typing import Tuple


class SpectralClusteringWithNodeFeatures(nn.Module):
    def __init__(self, cluster_number: int, gamma: float = 1.0):
        """
        Spectral Clustering module, which performs spectral clustering on a similarity matrix
        and returns both node features and cluster assignments.

        :param cluster_number: number of clusters
        :param gamma: RBF kernel parameter for computing similarity matrix
        """
        super(SpectralClusteringWithNodeFeatures, self).__init__()
        self.cluster_number = cluster_number
        self.gamma = gamma

    def forward(self, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform spectral clustering on the input batch of data, and return node features
        and cluster assignments.

        :param batch: [batch_size, num_nodes, feature_dim] similarity matrix or feature matrix
        :return: Tuple of [batch_size, num_nodes, feature_dim] node features and
                 [batch_size, num_nodes] cluster assignments
        """
        batch_size, num_nodes, feature_dim = batch.shape

        all_assignments = []
        all_node_features = []

        for i in range(batch_size):
            similarity_matrix = rbf_kernel(batch[i].detach().cpu().numpy(), gamma=self.gamma)

            degree_matrix = torch.diag(torch.sum(torch.tensor(similarity_matrix), dim=1))
            laplacian_matrix = degree_matrix - torch.tensor(similarity_matrix)

            eigvals, eigvecs = torch.linalg.eigh(laplacian_matrix)
            eigvecs = eigvecs[:, :self.cluster_number]

            node_features = torch.tensor(eigvecs).to(batch.device)
            all_node_features.append(node_features)

            kmeans = KMeans(n_clusters=self.cluster_number)
            kmeans.fit(node_features.detach().cpu().numpy())
            assignments = torch.tensor(kmeans.labels_).to(batch.device)
            all_assignments.append(assignments)

        # 将所有 batch 的节点特征和聚类结果拼接
        all_node_features = torch.stack(all_node_features, dim=0)  # [batch_size, num_nodes, feature_dim]
        all_assignments = torch.stack(all_assignments, dim=0)  # [batch_size, num_nodes]

        return all_node_features, all_assignments

    def loss(self, assignments, target_distribution):
        """
        Loss function can be modified for spectral clustering, but in basic form,
        spectral clustering does not have a direct loss function like DEC.
        You can use target distribution to modify the assignments.
        """
        return F.mse_loss(assignments, target_distribution)



batch_size = 2
num_nodes = 116
feature_dim = 116


random_data = torch.rand(batch_size, num_nodes, feature_dim)


model = SpectralClusteringWithNodeFeatures(cluster_number=5, gamma=1.0)


node_features, assignments = model(random_data)


print("Node Features Shape:", node_features.shape)
print("Cluster Assignments Shape:", assignments.shape)

print("Cluster Assignments for first batch:", assignments[0])
