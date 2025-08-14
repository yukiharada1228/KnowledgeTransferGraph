import torch
import torch.nn as nn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from torch.utils.data import DataLoader
import numpy as np


def accuracy(
    output: torch.Tensor, target: torch.Tensor, topk: tuple[int,] = (1,)
) -> list[torch.Tensor]:
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(100 * correct_k / batch_size)
    return res


class KNNValidation(nn.Module):
    def __init__(self, model, train_dataset, test_dataset, K=20):
        super(KNNValidation, self).__init__()
        self.model = model
        self.K = K
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=512,
            shuffle=False,
            num_workers=10,
            pin_memory=True,
            drop_last=False,
        )
        self.test_dataloader = DataLoader(
            test_dataset,
            batch_size=512,
            shuffle=False,
            num_workers=10,
            pin_memory=True,
            drop_last=False,
        )

    def forward(self) -> torch.Tensor:
        train_coords, train_labels = self.extract_features(self.train_dataloader)
        test_coords, test_labels = self.extract_features(self.test_dataloader)
        knn_acc = self.knn_classifier(
            train_coords, train_labels, test_coords, test_labels, k=self.K
        )
        return knn_acc

    def extract_features(self, dataloader):
        self.model.eval()

        coords = None
        labels = torch.zeros(len(dataloader.dataset))
        cnt = 0

        for inputs, targets in dataloader:
            inputs = inputs.cuda()
            batch_size = inputs.size(0)

            with torch.no_grad():
                features = self.model.encoder_features(inputs)

            if coords == None:
                coords = torch.zeros((len(dataloader.dataset), features.shape[1]))

            coords[cnt : cnt + batch_size] = features.data.cpu()
            labels[cnt : cnt + batch_size] = targets.data.cpu()
            cnt += batch_size
        return coords, labels

    def knn_classifier(self, train_coords, train_labels, test_coords, test_labels, k=1):
        classifier = KNeighborsClassifier(algorithm="brute", n_neighbors=k, n_jobs=-1)
        classifier.fit(train_coords.numpy(), train_labels.numpy())
        pred_test = classifier.predict(test_coords.numpy())
        knn_acc = torch.tensor(
            sum(pred_test == test_labels.numpy()) / len(pred_test) * 100
        )
        return knn_acc


class SilhouetteValidation(nn.Module):
    def __init__(
        self,
        model,
        train_dataset,
        test_dataset,
        eps: float = 0.5,
        min_samples: int = 5,
        metric: str = "euclidean",
    ):
        super(SilhouetteValidation, self).__init__()
        self.model = model
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.train_dataset = train_dataset  # 互換性維持のため保持（未使用）
        self.test_dataset = test_dataset
        self.test_dataloader = DataLoader(
            test_dataset,
            batch_size=512,
            shuffle=False,
            num_workers=10,
            pin_memory=True,
            drop_last=False,
        )

    def forward(self) -> torch.Tensor:
        # 評価はテスト埋め込みに対してDBSCANを適用し、そのクラスタでシルエット係数を算出
        test_coords = self.extract_features(self.test_dataloader)
        if test_coords is None:
            return torch.tensor(0.0)

        X = test_coords.numpy()

        clustering = DBSCAN(
            eps=self.eps, min_samples=self.min_samples, metric=self.metric
        )
        labels = clustering.fit_predict(X)

        # ノイズ(-1)を除外して評価（少なくとも2クラスタ必要）
        mask = labels != -1
        if mask.sum() < 2:
            return torch.tensor(0.0)
        labels_f = labels[mask]
        if len(set(labels_f.tolist())) < 2:
            return torch.tensor(0.0)
        score = silhouette_score(X[mask], labels_f, metric=self.metric)
        return torch.tensor(float(score))

    def extract_features(self, dataloader):
        self.model.eval()

        coords = None
        cnt = 0

        for inputs, _ in dataloader:
            inputs = inputs.cuda()
            batch_size = inputs.size(0)

            with torch.no_grad():
                features = self.model.encoder_features(inputs)

            if coords == None:
                coords = torch.zeros((len(dataloader.dataset), features.shape[1]))

            coords[cnt : cnt + batch_size] = features.data.cpu()
            cnt += batch_size
        return coords


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val: float = 0.0
        self.avg: float = 0.0
        self.sum: float = 0.0
        self.count: int = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
