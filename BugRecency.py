import pickle
import json
import os
from sklearn import preprocessing
import numpy as np
from Datasets import zxing, aspectj, swt
from datetime import datetime


def bugFixingRecency(ReportOpendate, fileFixingDate):
    if ReportOpendate is None or fileFixingDate is None:
        return 0

    else:
        return 1 / float(getMonthsBetween(ReportOpendate, fileFixingDate) + 1)


def getMonthsBetween(date1, date2):
    return abs((convertToDateTime(date1).year - convertToDateTime(date2).year) * 12 + convertToDateTime(date1).month - convertToDateTime(date2).month)


def convertToDateTime(date):
    return datetime.strptime(date, "%Y-%m-%d %H:%M:%S")


def runBugHistory(bugReports, srcFiles, currentDataset):
    fixedFiles = []
    scores = []
    i = 1
    for br in bugReports.values():

        for fixed in br.fixedFiles:
            fixedFiles.append(fixed)

        for src, value in srcFiles.items():
            if src in fixedFiles:
                value.srcFixedDate = br.fixedTime

        # for value in srcFiles.values():
        # print(value.srcFixedDate)
    with open(currentDataset.root + '/bugRecency.json', 'w') as file:
        for bugRep in bugReports.values():
            total_recency = []
            for src in srcFiles.values():
                if src in bugRep.fixedFiles:
                    reccency = 0
                else:
                    reccency = bugFixingRecency(bugRep.openDate, src.srcFixedDate)

                total_recency.append(reccency)
            scores.append(total_recency)
        json.dump(scores, file)

def main(data_Set):
    currentDataset = data_Set
    print("Bug history started")

    with open(currentDataset.root + '/preprocessed_reports.pickle', 'rb') as file:
        bug_reports = pickle.load(file)

    with open(currentDataset.root + '/preprocessed_src.pickle', 'rb') as file:
        src_files = pickle.load(file)

    runBugHistory(bug_reports, src_files, currentDataset)
    print('Bug history finished')


if __name__ == '__main__':
    main()
