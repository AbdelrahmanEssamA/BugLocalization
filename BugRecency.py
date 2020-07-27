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


def runBugHistory(bugReports, srcFiles):
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
        #    print(value.srcFixedDate)
    total_recency = []
    for bx in bugReports.values():
        total_recency.clear()
        for src in srcFiles.values():
            reccency = bugFixingRecency(bx.openDate, src.srcFixedDate)
            total_recency.append(reccency)
        scores.append(total_recency)
    return scores


def main():
    print("Bug history started")

    with open(aspectj.root + '/preprocessed_reports.pickle', 'rb') as file:
        bug_reports = pickle.load(file)

    with open(aspectj.root + '/preprocessed_src.pickle', 'rb') as file:
        src_files = pickle.load(file)

    bugRecency = runBugHistory(bug_reports, src_files)
    with open(aspectj.root + '/bugRecency.json', 'w') as file:
        json.dump(bugRecency, file)

    print('Bug history finished')


if __name__ == '__main__':
    main()

# most recent
#
