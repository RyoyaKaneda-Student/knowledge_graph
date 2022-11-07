import torch
from ignite.metrics import Metric
from ignite.exceptions import NotComputableError

# These decorators help with distributed settings
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced

from utils.torch import ZERO_FLOAT32_TENSOR


class RankingMetric(Metric):
    def __init__(self, output_transform=lambda x: x, device="cpu"):
        self._ranks: list[torch.Tensor] = []
        self._ZERO: torch.Tensor = ZERO_FLOAT32_TENSOR.detach().to(device)
        super(RankingMetric, self).__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self):
        self._ranks = []
        super(RankingMetric, self).reset()

    @reinit__is_reduced
    def update(self, output):
        pred, (row, column), filter_ = output
        assert pred.shape == filter_.shape
        zero, batch_size = self._ZERO, len(pred)
        pred = pred[row]
        filter_ = filter_[row]
        row, column = torch.arange(len(row)), column
        tmp = pred[row, column]
        pred[filter_] = zero
        pred[row, column] = tmp
        pred_list = torch.split(pred, batch_size, dim=0)
        column_list = torch.split(column, batch_size, dim=0)
        row_list = [row[:len(c)] for c in column_list]
        tmp_list: list[torch.Tensor] = [
            torch.argsort(torch.argsort(p, dim=1, descending=True), dim=1)[r, c] + 1
            for p, r, c in zip(pred_list, row_list, column_list)
        ]
        self._ranks.extend(tmp_list)

    @sync_all_reduce()
    def compute(self):
        ranks_list = self._ranks
        if len(ranks_list) == 0:
            raise NotComputableError("error.")
        ranks = torch.cat(ranks_list, dim=0)
        len_ranks = len(ranks)
        # mrr and hit
        mrr = (1. / ranks).sum() / len_ranks
        hit_ = [torch.count_nonzero(ranks <= (i + 1)) / len_ranks for i in range(10)]
        avg = ranks.sum()/len_ranks
        return {'ranks': ranks, 'mrr': mrr, 'hit_': hit_, 'avg': avg}
