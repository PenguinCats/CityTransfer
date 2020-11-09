import torch
import numpy as np


def _convert_rank_to_score(rel, pred):
    rel_rank_score = np.arange(1, len(rel) + 1)[::-1]
    rel2score = {pos: sc for pos, sc in zip(rel, rel_rank_score)}
    pred_rank_score = [rel2score.get(pos, 0.0) for pos in pred]
    return rel_rank_score, pred_rank_score


def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    if len(r):
        return np.sum(r / np.log2(np.arange(2, r.size+2)))
    else:
        return 0


def ndcg_at_k(rel, pred, k):
    rel = rel[:k]
    pred = pred[:k]
    rel_sc, pred_sc = _convert_rank_to_score(rel, pred)
    idcg = dcg_at_k(rel_sc, k)
    dcg = dcg_at_k(pred_sc, k)
    return dcg / idcg


if __name__ == '__main__':
    a = [3, 9, 2, 5]
    aaa = [3, 2, 3, 3, 2]
    # print(ndcg_at_k(a, b, 4))
    # print(ndcg_at_k(a, aa, 4))
    # print(ndcg_at_k(a, aaa, 4))
    print(torch.argsort(torch.Tensor(a)))
    print(torch.argsort(torch.Tensor(aaa)))