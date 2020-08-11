import spacy
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle
import json
from Datasets import aspectj, zxing, swt


class Word2Vec:
    # Empty  Constructor
    def __init__(self):
        pass

    def getSematicSemilarity(self, srcFiles, bugReport):
        # Load the glove english pretrained cnn
        nlp = spacy.load('en_core_web_lg')

        # Pass the src file tokens to glove
        srcDocs = [nlp(' '.join(src.fileName['lemma'] + src.classNames['lemma'] + src.attributes['lemma'] + src.comments['lemma'] + src.methodNames['lemma']))
                   for src in srcFiles.values()]
        mimMax = MinMaxScaler()
        simTable = []

        # Iterating on all bug reports
        for report in bugReport.values():
            # Pass the src file tokens to glove
            report_doc = nlp(' '.join(report.summary['lemma'] + report.pos_tagged_description['lemma']))
            scores = []

            for srcDoc in srcDocs:
                # calculate the similarity between each srcfile with the bug report
                similarity = report_doc.similarity(srcDoc)
                # Append the score
                scores.append(similarity)

            scores = np.array([float(count) for count in scores]).reshape(-1, 1)
           # print (scores)
            normalizedScores = np.concatenate(mimMax.fit_transform(scores))

            simTable.append(normalizedScores.tolist())

        return simTable


def main(data_Set):
    print("word2vec started")
    currentDataset = data_Set
    with open(currentDataset.root + '/preprocessed_src.pickle', 'rb') as file:
        srcFiles = pickle.load(file)

    with open(currentDataset.root + '/preprocessed_reports.pickle', 'rb') as file:
        bugReport = pickle.load(file)

    data = Word2Vec()
    simTable = data.getSematicSemilarity(srcFiles, bugReport)
    with open(currentDataset.root + '/semantic_similarity.json', 'w') as file:
        json.dump(simTable, file)

    print('Glove component executed successfully')


if __name__ == '__main__':
    main()
