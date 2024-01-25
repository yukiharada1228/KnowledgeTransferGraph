import random

import torch


def set_seed(manualSeed: int):
    # Fix the seed value
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    # if True, causes cuDNN to benchmark multiple convolution algorithms and select the fastest.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = (
        True  # if True, causes cuDNN to only use deterministic convolution algorithms.
    )
    torch.backends.cudnn.benchmark = False  # if True, causes cuDNN to benchmark multiple convolution algorithms and select the fastest.


class WorkerInitializer:
    def __init__(self, manualSeed: int):
        self.manualSeed = manualSeed

    def worker_init_fn(self, worker_id: int):
        random.seed(self.manualSeed + worker_id)
