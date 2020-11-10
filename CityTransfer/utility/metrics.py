import torch
import numpy as np
from CityTransfer.utility.log_helper import logging


def dcg_score(score, rank):
    r = np.asfarray(score)
    if len(r):
        rank = [v+1 for v in rank]
        return torch.sum(score / np.log2(rank))
    else:
        return 0


def ndcf_at_k(rel_score, pred_score, k):
    sorted_rel_score, rel_rank = torch.sort(rel_score, descending=True)
    _, pred_rank = torch.sort(pred_score, descending=True)
    valid_len = min(len(torch.nonzero(sorted_rel_score)), k)

    rel_rank = rel_rank.numpy()
    pred_rank = pred_rank.numpy()

    rel2score = {pos: sc for pos, sc in zip(rel_rank, sorted_rel_score)}
    rel2rank = {pos: sc+1 for pos, sc in zip(rel_rank, range(len(rel_rank)))}

    pred_corresponding_score = torch.Tensor([rel2score[pos] for pos in pred_rank])
    pred_corresponding_rank = torch.Tensor([rel2rank[pos] for pos in pred_rank])

    real_corresponding_score = sorted_rel_score[:valid_len]
    real_corresponding_rank = [rk + 1 for rk in range(len(real_corresponding_score))]
    pred_corresponding_score = pred_corresponding_score[:valid_len]
    pred_corresponding_rank = pred_corresponding_rank[:valid_len]

    idcg = dcg_score(real_corresponding_score, real_corresponding_rank)
    dcg = dcg_score(pred_corresponding_score, pred_corresponding_rank)

    return dcg / idcg


def ndcf_at_k_test(rel_score, pred_score, k):
    sorted_rel_score, rel_rank = torch.sort(rel_score, descending=True)
    _, pred_rank = torch.sort(pred_score, descending=True)
    valid_len = min(len(torch.nonzero(sorted_rel_score)), k)

    rel_rank = rel_rank.numpy()
    pred_rank = pred_rank.numpy()

    rel2score = {pos: sc for pos, sc in zip(rel_rank, sorted_rel_score)}
    rel2rank = {pos: sc+1 for pos, sc in zip(rel_rank, range(len(rel_rank)))}

    pred_corresponding_score = torch.Tensor([rel2score[pos] for pos in pred_rank])
    pred_corresponding_rank = torch.Tensor([rel2rank[pos] for pos in pred_rank])

    real_corresponding_score = sorted_rel_score[:valid_len]
    real_corresponding_rank = [rk + 1 for rk in range(len(real_corresponding_score))]
    pred_corresponding_score = pred_corresponding_score[:valid_len]
    pred_corresponding_rank = pred_corresponding_rank[:valid_len]

    logging.info("Real Rank:")
    logging.info(rel_rank[:valid_len])
    logging.info("Pred Rank:")
    logging.info(pred_rank[:valid_len*2])

    idcg = dcg_score(real_corresponding_score, real_corresponding_rank)
    dcg = dcg_score(pred_corresponding_score, pred_corresponding_rank)

    return dcg / idcg,  rel_rank[:valid_len], pred_rank[:valid_len], pred_rank[valid_len:valid_len*2]
