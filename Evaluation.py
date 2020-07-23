import pickle
import json
import os
import operator
import numpy as np
from Datasets import aspectj, swt, zxing
from scipy import optimize


def combine_rank_scores(coeffs, *rank_scores):
    final_score = []
    for scores in zip(*rank_scores):
        combined_score = coeffs @ np.array(scores)
        final_score.append(combined_score)

    return final_score


def cost(coeffs, src_files, bug_reports, *rank_scores):
    final_scores = combine_rank_scores(coeffs, *rank_scores)
    mrr = []
    mean_avgp = []

    for i, report in enumerate(bug_reports.items()):
        src_ranks, _ = zip(*sorted(zip(src_files.keys(), final_scores[i]),
                                   key=operator.itemgetter(1), reverse=True))

        # Getting reported fixed files
        fixed_files = report[1].fixedFiles

        # Getting the ranks of reported fixed files
        relevant_ranks = sorted(src_ranks.index(fixed) + 1 for fixed in fixed_files)
        # MRR
        min_rank = relevant_ranks[0]
        mrr.append(1 / min_rank)

        # MAP
        mean_avgp.append(np.mean([len(relevant_ranks[:j + 1]) / rank for j, rank in enumerate(relevant_ranks)]))

    return -1 * (np.mean(mrr) + np.mean(mean_avgp))


def estiamte_params(src_files, bug_reports, *rank_scores):
    res = optimize.differential_evolution(
        cost, bounds=[(0, 1)] * len(rank_scores),
        args=(src_files, bug_reports, *rank_scores),
        strategy='randtobest1exp', polish=True, seed=458711526
    )

    return res.x.tolist()


def evaluate(src_files, bug_reports, coeffs, *rank_scores):
    final_scores = combine_rank_scores(coeffs, *rank_scores)

    top_n = (1, 5, 10)
    top_n_rank = [0] * len(top_n)
    mrr = []
    mean_avgp = []

    precision_at_n = [[] for _ in top_n]
    recall_at_n = [[] for _ in top_n]
    f_measure_at_n = [[] for _ in top_n]

    for i, (bug_id, report) in enumerate(bug_reports.items()):

        # Finding source codes from the simis indices
        src_ranks, _ = zip(*sorted(zip(src_files.keys(), final_scores[i]),
                                   key=operator.itemgetter(1), reverse=True))

        # Getting reported fixed files
        fixed_files = report.fixedFiles

        # Iterating over top n
        for k, rank in enumerate(top_n):
            hit = set(src_ranks[:rank]) & set(fixed_files)

            # Computing top n rank
            if hit:
                top_n_rank[k] += 1

            # Computing precision and recall at n
            if not hit:
                precision_at_n[k].append(0)
            else:
                precision_at_n[k].append(len(hit) / len(src_ranks[:rank]))
            recall_at_n[k].append(len(hit) / len(fixed_files))
            if not (precision_at_n[k][i] + recall_at_n[k][i]):
                f_measure_at_n[k].append(0)
            else:
                f_measure_at_n[k].append(2 * (precision_at_n[k][i] * recall_at_n[k][i])
                                         / (precision_at_n[k][i] + recall_at_n[k][i]))

        # Getting the ranks of reported fixed files
        relevant_ranks = sorted(src_ranks.index(fixed) + 1
                                for fixed in fixed_files)
        # MRR
        min_rank = relevant_ranks[0]
        mrr.append(1 / min_rank)

        # MAP
        mean_avgp.append(np.mean([len(relevant_ranks[:j + 1]) / rank for j, rank in enumerate(relevant_ranks)]))

    return (top_n_rank, [x / len(bug_reports) for x in top_n_rank],
            np.mean(mrr), np.mean(mean_avgp),
            np.mean(precision_at_n, axis=1).tolist(), np.mean(recall_at_n, axis=1).tolist(),
            np.mean(f_measure_at_n, axis=1).tolist())


def main():
    with open(zxing.root + '/preprocessed_src.pickle', 'rb') as file:
        src_files = pickle.load(file)
    with open(zxing.root + '/preprocessed_reports.pickle', 'rb') as file:
        bug_reports = pickle.load(file)

    with open(zxing.root + '/similarities4.json', 'r') as file:
        gensim = json.load(file)

    params = estiamte_params(src_files, bug_reports, gensim)
    results = evaluate(src_files, bug_reports, params, gensim)

    print('Top N Rank:', results[0])
    print('Top 1 Rank %:', results[1][0])
    print('Top 5 Rank %:', results[1][1])
    print('Top 10 Rank %:', results[1][2])
    print('MRR:', results[2])
    print('MAP:', results[3])

    # Uncomment these for precision, recall, and f-measure results
    # print('Precision@N:', results[4])
    # print('Recall@N:', results[5])
    # print('F-measure@N:', results[6])


if __name__ == '__main__':
    main()
