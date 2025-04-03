import torch


class PredictionMetrics:
    """ADE metric in pytorch"""

    def __init__(self):
        super().__init__()
        self.ades = []
        self.fdes = []
        self.total = 0

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        if preds.shape != target.shape:
            raise ValueError("Problem on ADE shapes")
        displacement_error = torch.norm(preds - target, p=2, dim=-1)

        self.ades.append(displacement_error.mean(-1).sum())
        self.fdes.append(displacement_error[:, -1].sum())
        self.total += target.shape[0]  # evaluating bs

    def compute(self):
        ade = sum(self.ades) / self.total
        fde = sum(self.fdes) / self.total
        return ade.item(), fde.item()

    def reset(self):
        self.ades.clear()
        self.fdes.clear()
        self.total = 0
