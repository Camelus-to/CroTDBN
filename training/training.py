from source.utils import accuracy, TotalMeter, count_params, isfloat
import torch
import numpy as np
from pathlib import Path
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support, classification_report
from source.utils import continus_mixup_data
import wandb
from omegaconf import DictConfig
from typing import List
import torch.utils.data as utils
from source.components import LRScheduler
import logging


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, verbose=False, path='checkpoint.pt'):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.best_score = None
        self.epochs_no_improve = 0
        self.early_stop = False
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.min_delta:
            self.epochs_no_improve += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.epochs_no_improve}/{self.patience}')
            if self.epochs_no_improve >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.epochs_no_improve = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.best_score:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), self.path)


class Train:
    def __init__(self, cfg: DictConfig,
                 model: torch.nn.Module,
                 optimizers: List[torch.optim.Optimizer],
                 lr_schedulers: List[LRScheduler],
                 dataloaders: List[utils.DataLoader],
                 logger: logging.Logger,
                 semantic_similarity_matrix: torch.Tensor) -> None:

        self.config = cfg
        self.logger = logger
        self.model = model
        self.logger.info(f'#model params: {count_params(self.model)}')
        self.semantic_similarity_matrix = semantic_similarity_matrix
        self.train_dataloader, self.val_dataloader, self.test_dataloader = dataloaders
        self.epochs = cfg.training.epochs
        self.total_steps = cfg.total_steps
        self.optimizers = optimizers
        self.lr_schedulers = lr_schedulers
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
        self.save_path = Path(cfg.log_path) / cfg.unique_id
        self.save_learnable_graph = cfg.save_learnable_graph

        self.init_meters()

    def init_meters(self):
        self.train_loss, self.val_loss, \
            self.test_loss, self.train_accuracy, \
            self.val_accuracy, self.test_accuracy = [
            TotalMeter() for _ in range(6)]

    def reset_meters(self):
        for meter in [self.train_accuracy, self.val_accuracy,
                      self.test_accuracy, self.train_loss,
                      self.val_loss, self.test_loss]:
            meter.reset()

    def train_per_epoch(self, optimizer, lr_scheduler):
        self.model.train()
        semantic_similarity_matrix = self.semantic_similarity_matrix
        for time_series, node_feature, dynamic_fc, label in self.train_dataloader:
            label = label.float()
            self.current_step += 1

            lr_scheduler.update(optimizer=optimizer, step=self.current_step)
            time_series, node_feature, label = time_series, node_feature, label
            if self.config.preprocess.continus:
                time_series, node_feature, dynamic_fc, label = continus_mixup_data(
                    time_series, node_feature, dynamic_fc, y=label)

            predict, contra_loss = self.model(time_series, node_feature, dynamic_fc, semantic_similarity_matrix)
            alpha = 0.6
            loss = self.loss_fn(predict, label)
            loss = loss + alpha * contra_loss
            self.train_loss.update_with_weight(loss.item(), label.shape[0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            top1 = accuracy(predict, label[:, 1])[0]
            self.train_accuracy.update_with_weight(top1, label.shape[0])


    def test_per_epoch(self, dataloader, loss_meter, acc_meter):
        labels = []
        result = []

        self.model.eval()
        semantic_similarity_matrix = self.semantic_similarity_matrix
        for time_series, node_feature, dynamic_fc, label in dataloader:
            time_series, node_feature, label = time_series, node_feature, label
            output, co_loss = self.model(time_series, node_feature, dynamic_fc, semantic_similarity_matrix)

            label = label.float()

            loss = self.loss_fn(output, label)
            loss = loss
            loss_meter.update_with_weight(
                loss.item(), label.shape[0])
            top1 = accuracy(output, label[:, 1])[0]
            acc_meter.update_with_weight(top1, label.shape[0])
            result += F.softmax(output, dim=1)[:, 1].tolist()
            labels += label[:, 1].tolist()

        auc = roc_auc_score(labels, result)

        result, labels = np.array(result), np.array(labels)
        result[result > 0.5] = 1
        result[result <= 0.5] = 0
        metric = precision_recall_fscore_support(
            labels, result, average='micro')

        report = classification_report(
            labels, result, output_dict=True, zero_division=0)

        recall = [0, 0]
        for k in report:
            if isfloat(k):
                recall[int(float(k))] = report[k]['recall']
        return [auc] + list(metric) + recall

    def generate_save_learnable_matrix(self):


        learable_matrixs = []

        labels = []

        for time_series, node_feature, label in self.test_dataloader:
            label = label.long()

            time_series, node_feature, label = time_series, node_feature, label
            _, learable_matrix, _ = self.model(time_series, node_feature)

            learable_matrixs.append(learable_matrix.cpu().detach().numpy())
            labels += label.tolist()

        self.save_path.mkdir(exist_ok=True, parents=True)
        np.save(self.save_path / "learnable_matrix.npy", {'matrix': np.vstack(
            learable_matrixs), "label": np.array(labels)}, allow_pickle=True)

    def save_result(self, results: torch.Tensor):
        self.save_path.mkdir(exist_ok=True, parents=True)
        np.save(self.save_path / "training_process.npy",
                results, allow_pickle=True)

        torch.save(self.model.state_dict(), self.save_path / "model.pt")

    def train(self):
        training_process = []
        early_stopping = EarlyStopping(patience=5, verbose=True, path=self.save_path / "best_model.pt")  # 设定早停策略
        self.current_step = 0
        for epoch in range(self.epochs):  # epochs是200轮
            self.reset_meters()
            self.train_per_epoch(self.optimizers[0], self.lr_schedulers[0])  # 单独训练每一轮
            val_result = self.test_per_epoch(self.val_dataloader,
                                             self.val_loss, self.val_accuracy)

            test_result = self.test_per_epoch(self.test_dataloader,
                                              self.test_loss, self.test_accuracy)

            self.logger.info(" | ".join([
                f'Lunshu[{self.config.lunshu}]',
                f'Epoch[{epoch}/{self.epochs}]',
                f'Train Loss:{self.train_loss.avg: .3f}',
                f'Train Accuracy:{self.train_accuracy.avg: .3f}%',
                f'Test Loss:{self.test_loss.avg: .3f}',
                f'Test Accuracy:{self.test_accuracy.avg: .3f}%',
                f'Val AUC:{val_result[0]:.4f}',
                f'Test AUC:{test_result[0]:.4f}',
                f'Test Precision:{test_result[1]:.4f}',
                f'Test Recall:{test_result[2]:.4f}',
                f'Test F1-score:{test_result[3]:.4f}',
                f'Test Sen:{test_result[-1]:.4f}',
                f'LR:{self.lr_schedulers[0].lr:.4f}'
            ]))


            training_process.append({
                "Epoch": epoch,
                "Train Loss": self.train_loss.avg,
                "Train Accuracy": self.train_accuracy.avg,
                "Test Loss": self.test_loss.avg,
                "Test Accuracy": self.test_accuracy.avg,
                "Test AUC": test_result[0],
                'Test Sensitivity': test_result[-1],
                'Test Specificity': test_result[-2],
                'micro F1': test_result[-4],
                'micro recall': test_result[-5],
                'micro precision': test_result[-6],
                "Val AUC": val_result[0],
                "Val Loss": self.val_loss.avg,
            })

            early_stopping(self.val_loss.avg, self.model)

            if early_stopping.early_stop:
                self.logger.info(f"Early stopping at epoch {epoch}")
                break

        if self.save_learnable_graph:
            self.generate_save_learnable_matrix()
        self.save_result(training_process)
