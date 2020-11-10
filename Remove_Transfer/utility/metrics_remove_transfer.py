import torch
import numpy as np


def dcg_score(score):
    upper = (2**score) - 1
    down = torch.log2(torch.Tensor([i + 2 for i in np.arange(score.shape[0])]))
    return torch.sum(upper / down)


def ndcf_at_k(rel_score, predict_score, k):
    sorted_rel_score, rel_rank = torch.sort(rel_score, descending=True)
    _, pred_rank = torch.sort(predict_score, descending=True)

    exist_shop_n = len(torch.nonzero(sorted_rel_score))
    valid_len = min(len(torch.nonzero(sorted_rel_score)), k)
    exist_shop_grid_id = rel_rank[:exist_shop_n]

    real_corresponding_score = [1 for _ in range(valid_len)]
    while len(real_corresponding_score) < k:
        real_corresponding_score.append(0)
    pred_corresponding_score = [1 if grid_id in exist_shop_grid_id else 0 for grid_id in pred_rank[:k]]

    real_corresponding_score = torch.Tensor(real_corresponding_score)
    pred_corresponding_score = torch.Tensor(pred_corresponding_score)

    idcg = dcg_score(real_corresponding_score)
    dcg = dcg_score(pred_corresponding_score)

    return dcg / idcg


def ndcf_at_k_test(rel_score, pred_score, k):
    sorted_rel_score, rel_rank = torch.sort(rel_score, descending=True)
    _, pred_rank = torch.sort(pred_score, descending=True)

    exist_shop_n = len(torch.nonzero(sorted_rel_score))
    valid_len = min(exist_shop_n, k)
    exist_shop_grid_id = rel_rank[:exist_shop_n]

    real_corresponding_score = [1 for _ in range(valid_len)]
    while len(real_corresponding_score) < k:
        real_corresponding_score.append(0)
    pred_corresponding_score = [1 if grid_id in exist_shop_grid_id else 0 for grid_id in pred_rank[:k]]

    real_corresponding_score = torch.Tensor(real_corresponding_score)
    pred_corresponding_score = torch.Tensor(pred_corresponding_score)

    idcg = dcg_score(real_corresponding_score)
    dcg = dcg_score(pred_corresponding_score)

    return dcg / idcg, exist_shop_grid_id, pred_rank[:k], pred_rank[k:k*2]
