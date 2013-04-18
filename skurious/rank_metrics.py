
import numpy as np


def delta_avg(preds, refs):
    data = zip(preds, refs)
    data.sort(key=lambda x: x[0], reverse=True)
    rank = np.array([d[1] for d in data])
    avg = np.mean(rank)
    N = int(len(rank) / 2)
    total = 0
    for n in range(2,N+1):
        total += delta_avg_n(rank, n, avg)
    return total / float(N - 1)
   

def delta_avg_n(rank, n, avg):
    q_size = int(len(rank) / n)
    denom = float(n - 1)
    num = 0
    for i in range(n - 1):
        quantil = rank[:(i+1)*q_size]
        num += np.mean(quantil)
    return (num / denom) - avg


def adjust_rank(scores, rank):
    i = 0
    while i < len(scores) - 1:
        j = i + 1
        if scores[i] == scores[j]:
            rank_sum = rank[i]
            n = 1. #float cast
            while (j < len(scores) and (scores[i] == scores[j])):
                rank_sum += rank[j]
                n += 1
                j += 1
            while i < j:
                rank[i] = rank_sum / n
                i += 1
        i += 1
    return rank


def spearman(X0, X1):
    n = len(X0)
    N = np.array([range(1, n + 1)], dtype=float).T
    # sort first rank
    data = np.array([X0, X1]).T
    data = data[data[:, 0].argsort()[::-1]]
    data = np.concatenate((data, N), axis=1).T
    data[2] = adjust_rank(data[0], data[2])
    # sort second rank
    data = data.T
    data = data[data[:, 1].argsort()[::-1]]
    data = np.concatenate((data, N), axis=1).T
    data[3] = adjust_rank(data[1], data[3])
    # calculate spearman
    num = 6 * sum((data[2] - data[3]) ** 2)
    denom = n * (n ** 2 - 1)
    return 1 - (num / float(denom))
