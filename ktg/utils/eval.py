import torch
import torch.nn as nn
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader
from .seed import WorkerInitializer


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
