import os

import torch


def save_checkpoint(model, save_dir, epoch, is_best=False):
    state = {
        "epoch": epoch,
        "arch": model.__class__.__name__,
        "model_pram": model.state_dict(),
    }
    if is_best:
        path = os.path.join(save_dir, "best_checkpoint.pkl")
    else:
        path = os.path.join(save_dir, "checkpoint_epoch_%d.pkl" % epoch)
    torch.save(state, path, pickle_protocol=4)


def load_checkpoint(model, save_dir, epoch=1, is_best=False):
    if is_best:
        path = os.path.join(save_dir, "best_checkpoint.pkl")
    else:
        path = os.path.join(save_dir, "checkpoint_epoch_%d.pkl" % epoch)
    # PyTorch 2.6 以降の既定 (weights_only=True) により読み込みが失敗するため明示的に False を指定
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.cpu()
    model.load_state_dict(state["model_pram"])
    model.cuda()
