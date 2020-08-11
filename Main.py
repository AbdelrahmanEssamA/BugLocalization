import Datasets
import BugReportPreprocessing
import SourceCodePreprocessing
import rVSM
import WordEmbedding
import TokenMatch
import BugRecency
import Evaluation

data_set = Datasets.zxing
BugReportPreprocessing.main(data_set)
SourceCodePreprocessing.main(data_set)
rVSM.main(data_set)
TokenMatch.main(data_set)
WordEmbedding.main(data_set)
BugRecency.main(data_set)
print('---------------------------------------------------')
Evaluation.main(data_set)