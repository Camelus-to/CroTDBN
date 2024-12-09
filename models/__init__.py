import torch

from .transformer import GraphTransformer
from omegaconf import DictConfig

from .BNT import BrainNetworkTransformer
from .BNT import BrainNetworkTransformer_adni
from .Test import BrainNetworkTransformer_mdd
from .Test import BrainNetworkTransformer_abide
# from .Test import BrainNetworkTransformer_adni
from .BNT import SVM
from .BNT import GAT
from .BNT import Transformer_duibi
from .Xiaorong import BrainNetworkTransformer_no_dynamic
from .Xiaorong import BrainNetworkTransformer_no_contra_loss
from .Xiaorong import BrainNetworkTransformer_no_pos
from .Xiaorong import BrainNetworkTransformer_no_dynamic_contra
from .Xiaorong import BrainNetworkTransformer_no_pos_contra


def xiaorong_model_factory(config: DictConfig, semantic_similarity_matrix: torch.Tensor, leixing: str):
    if config.model.name in ["LogisticRegression", "SVC"]:
        return None
    # model_name是这个了 BrainNetworkTransformer
    model = leixing
    return eval(model)(config, semantic_similarity_matrix)


def test_model_factory(config: DictConfig, semantic_similarity_matrix: torch.Tensor, dataset: str):
    if config.model.name in ["LogisticRegression", "SVC"]:
        return None
    # model_name是这个了 BrainNetworkTransformer
    model = config.model.name + f'_{dataset}'
    return eval(model)(config, semantic_similarity_matrix)


def MachineLearning_model_factory(config: DictConfig, semantic_similarity_matrix: torch.Tensor):
    # model_name是这个了 BrainNetworkTransformer
    model = config.jiqixuexi
    return eval(model)(config, semantic_similarity_matrix)


def model_factory(config: DictConfig, semantic_similarity_matrix: torch.Tensor):
    if config.model.name in ["LogisticRegression", "SVC"]:
        return None
    # model_name是这个了 BrainNetworkTransformer
    model = config.model.name + f'_{config.which_model}'
    if config.duibifangfa == 'GAT':
        model = 'GAT'
    if config.duibifangfa == 'Transformer_duibi':
        model = 'Transformer_duibi'
    return eval(model)(config, semantic_similarity_matrix)
