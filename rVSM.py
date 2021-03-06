import pickle
import json
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from  Datasets import  zxing, aspectj, swt
#from gensim.matutils import softcossim


class VectorSpaceModel:

    def __init__(self, src_files):
        self.src_files = src_files
        self.src_strings = [' '.join(src.fileName['lemma'] + src.classNames['lemma'] + src.methodNames['lemma'] + src.comments['lemma'] + src.attributes['lemma'])
                            for src in self.src_files.values()]

    def calculateSimilarity(self, src_tfidf, reports_tfidf):

        # Normalizing the length of source files
        src_lenghts = np.array([float(len(src_str.split())) for src_str in self.src_strings]).reshape(-1, 1)
        min_max_scaler = preprocessing.MinMaxScaler()
        normalized_src_len = min_max_scaler.fit_transform(src_lenghts)

        # length score
        src_len_score = 1 / (1 + np.exp(-12 * normalized_src_len))

        simis = []
        for report in reports_tfidf:
            s = cosine_similarity(src_tfidf, report)
            # revised VSM score calculation cosine similarity x lengh score
            rvsm_score = s * src_len_score
            normalized_score = np.concatenate(min_max_scaler.fit_transform(rvsm_score))
            simis.append(normalized_score.tolist())

        return simis

    def findSimilars(self, bug_reports):

        reports_strings = [' '.join(report.summary['lemma'] + report.description['lemma']) for report in bug_reports.values()]
        tfidf = TfidfVectorizer(sublinear_tf=True, smooth_idf=False)
        #vectorization
        src_tfidf = tfidf.fit_transform(self.src_strings)
        reports_tfidf = tfidf.transform(reports_strings)
        simis = self.calculateSimilarity(src_tfidf, reports_tfidf)

        return simis


def main(data_Set):
    currentDataset = data_Set
    print("rvsm started")
    # Unpickle preprocessed data
    with open(currentDataset.root + '/preprocessed_src.pickle', 'rb') as file:
        src_files = pickle.load(file)

    with open(currentDataset.root + '/preprocessed_reports.pickle', 'rb') as file:
        bug_reports = pickle.load(file)

    sm = VectorSpaceModel(src_files)
    simis = sm.findSimilars(bug_reports)

    # Saving similarities in a json file
    with open(currentDataset.root + '/vsm_similarity.json', 'w') as file:
        json.dump(simis, file)
    print('done')

if __name__ == '__main__':
    main()
