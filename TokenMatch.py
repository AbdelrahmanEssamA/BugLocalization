import pickle
import json
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from  Datasets import  zxing, aspectj, swt


def check_matchings(src_files, bug_reports):
    """Checking the matching tokens between bug reports and source files"""

    scores = []
    for report in bug_reports.values():
        matched_count = []
        summary_set = report.summary
        pos_tagged_sum_desc = (report.pos_tagged_summary['lemmatize'] + report.pos_tagged_description['lemmatize'])

        for src in src_files.values():
            if src.fileName['lemmatize']:
                common_tokens = len(set(summary_set['lemmatize']) & set([src.fileName['lemmatize'][0]]))

            matched_count.append(common_tokens)

        # Here no files matched a summary
        if sum(matched_count) == 0:
            matched_count = []
            for src in src_files.values():
                common_tokens = len(set(pos_tagged_sum_desc) & set(src.fileName['unu'] + src.classNames['lemmatize'] + src.methodNames['lemmatize']))

                if not common_tokens:
                    common_tokens = (len(set(pos_tagged_sum_desc) & set(src.comments['lemmatize'])) - len(set(src.comments['lemmatize'])))

                if not common_tokens:
                    common_tokens = (len(set(pos_tagged_sum_desc) & set(src.attributes['lemmatize'])) - len(set(src.attributes['lemmatize'])))

                matched_count.append(common_tokens)

        min_max_scaler = preprocessing.MinMaxScaler()

        intersect_count = np.array([float(count) for count in matched_count]).reshape(-1, 1)
        normalized_count = np.concatenate ( min_max_scaler.fit_transform(intersect_count))

        scores.append(normalized_count.tolist())

    return scores


def main():
    print("Token matching started")
    # Unpickle preprocessed data
    with open(zxing.root + '/preprocessed_src.pickle', 'rb') as file:
        src_files = pickle.load(file)

    with open(zxing.root + '/preprocessed_reports.pickle', 'rb') as file:
        bug_reports = pickle.load(file)

    scores = check_matchings(src_files,bug_reports)

    # Saving similarities in a json file
    with open(zxing.root + '/token_matching.json', 'w') as file:
        json.dump(scores, file)
    print('Token matching  finished')

if __name__ == '__main__':
    main()
