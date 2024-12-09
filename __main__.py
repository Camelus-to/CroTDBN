from datetime import datetime
import wandb
import hydra
from omegaconf import DictConfig, open_dict
from dataset import dataset_factory
from models import model_factory
from components import lr_scheduler_factory, optimizers_factory, logger_factory
from training import training_factory
from datetime import datetime
import pandas as pd
import torch

device = 'cpu'


def model_training(cfg: DictConfig):
    with open_dict(cfg):
        cfg.unique_id = datetime.now().strftime("%m-%d-%H-%M-%S")

    if cfg.dataset.name == 'abide':
        semantic_similarity_matrix = pd.read_csv(
            'E:\Technolgy_learning\Learning_code\AD\AD_Bert\获取文本\similarity_matrix.csv', header=None)

    elif cfg.dataset.name == 'adni':
        semantic_similarity_matrix = pd.read_csv(
            'E:\Technolgy_learning\Learning_code\AD\AD_Bert\获取文本\similarity_matrix.csv', header=None)

    elif cfg.dataset.name == 'mdd':
        semantic_similarity_matrix = pd.read_csv(
            'E:\Technolgy_learning\Learning_code\AD\AD_Bert\获取文本\similarity_matrix.csv', header=None)

    semantic_similarity_numpy = semantic_similarity_matrix.values  # DataFrame 转 NumPy 数组

    semantic_similarity_matrix = torch.tensor(semantic_similarity_numpy, dtype=torch.float32)
    semantic_similarity_matrix = semantic_similarity_matrix.to(device)

    dataloaders = dataset_factory(cfg)
    logger = logger_factory(cfg)
    model = model_factory(cfg, semantic_similarity_matrix)
    optimizers = optimizers_factory(
        model=model, optimizer_configs=cfg.optimizer)
    lr_schedulers = lr_scheduler_factory(lr_configs=cfg.optimizer,
                                         cfg=cfg)
    training = training_factory(cfg, model, optimizers,
                                lr_schedulers, dataloaders, logger, semantic_similarity_matrix)

    training.train()


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    for _ in range(cfg.repeat_time):
        cfg.lunshu = _
        model_training(cfg)


if __name__ == '__main__':
    main()
