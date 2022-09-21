import torch
import numpy as np
import datetime

from spodernet.utils.logger import Logger
from torch.autograd import Variable
from sklearn import metrics


def ranking_and_hits(model, dev_rank_batcher, vocab, name, *, log=None):
    def log_info(x):
        if log is not None: log.info(x)

    log_info('')
    log_info('-' * 50)
    log_info(name)
    log_info('-' * 50)
    log_info('')
    hits_left = []
    hits_right = []
    hits = []
    ranks = []
    ranks_left = []
    ranks_right = []
    for i in range(10):
        hits_left.append([])
        hits_right.append([])
        hits.append([])

    # loss = 0
    for i, str2var in enumerate(dev_rank_batcher):
        e1 = str2var['e1']
        e2 = str2var['e2']
        rel = str2var['rel']
        rel_reverse = str2var['rel_eval']
        e2_multi1 = str2var['e2_multi1'].float()
        e2_multi2 = str2var['e2_multi2'].float()
        pred1 = model.forward((e1, rel))
        pred2 = model.forward((e2, rel_reverse))
        pred1, pred2 = pred1.data, pred2.data
        e1, e2 = e1.data, e2.data
        e2_multi1, e2_multi2 = e2_multi1.data, e2_multi2.data
        for i in range(e1.shape[0]):
            # these filters contain ALL labels
            filter1 = e2_multi1[i].long()
            filter2 = e2_multi2[i].long()

            num = e1[i, 0].item()
            # save the prediction that is relevant
            target_value1 = pred1[i, e2[i, 0].item()].item()
            target_value2 = pred2[i, e1[i, 0].item()].item()
            # zero all known cases (this are not interesting)
            # this corresponds to the filtered setting
            pred1[i][filter1] = 0.0
            pred2[i][filter2] = 0.0
            # write base the saved values
            pred1[i][e2[i]] = target_value1
            pred2[i][e1[i]] = target_value2

        # sort and rank
        max_values, argsort1 = torch.sort(pred1, 1, descending=True)
        max_values, argsort2 = torch.sort(pred2, 1, descending=True)

        argsort1 = argsort1.cpu().numpy()
        argsort2 = argsort2.cpu().numpy()
        for i in range(e1.shape[0]):
            # find the rank of the target entities
            rank1 = np.where(argsort1[i] == e2[i, 0].item())[0][0]
            rank2 = np.where(argsort2[i] == e1[i, 0].item())[0][0]
            # rank+1, since the lowest rank is rank 1 not rank 0
            ranks.append(rank1 + 1)
            ranks_left.append(rank1 + 1)
            ranks.append(rank2 + 1)
            ranks_right.append(rank2 + 1)

            # this could be done more elegantly, but here you go
            for hits_level in range(10):
                if rank1 <= hits_level:
                    hits[hits_level].append(1.0)
                    hits_left[hits_level].append(1.0)
                else:
                    hits[hits_level].append(0.0)
                    hits_left[hits_level].append(0.0)

                if rank2 <= hits_level:
                    hits[hits_level].append(1.0)
                    hits_right[hits_level].append(1.0)
                else:
                    hits[hits_level].append(0.0)
                    hits_right[hits_level].append(0.0)

        dev_rank_batcher.state.loss = [0]

    for i in range(10):
        log_info('Hits left @{0}: {1}'.format(i + 1, np.mean(hits_left[i])))
        log_info('Hits right @{0}: {1}'.format(i + 1, np.mean(hits_right[i])))
        log_info('Hits @{0}: {1}'.format(i + 1, np.mean(hits[i])))

    mean_rank_left, mean_rank_right, mean_rank = np.mean(ranks_left), np.mean(ranks_right), np.mean(ranks)
    log_info('Mean rank left: {0}'.format(mean_rank_left))
    log_info('Mean rank right: {0}'.format(mean_rank_right))
    log_info('Mean rank: {0}'.format(mean_rank))
    mean_reciprocal_rank_left = np.mean(1. / np.array(ranks_left))
    mean_reciprocal_rank_right = np.mean(1. / np.array(ranks_right))
    mean_reciprocal_rank = np.mean(1. / np.array(ranks))
    log_info('Mean reciprocal rank left: {0}'.format(mean_reciprocal_rank_left))
    log_info('Mean reciprocal rank right: {0}'.format(mean_reciprocal_rank_right))
    log_info('Mean reciprocal rank: {0}'.format(mean_reciprocal_rank))
    return (mean_rank_left, mean_rank_right, mean_rank,
            mean_reciprocal_rank_left, mean_reciprocal_rank_right, mean_reciprocal_rank)
