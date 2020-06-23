import pickle
import json
import os
from sklearn import preprocessing
import numpy as np
from Datasets import zxing, aspectj, swt
from datetime import datetime


# 1. eh howa el buggy src_file, bug report bug reports
# buggy src file = ben loop 3la fixed files w nest5dm norm path
# bug_report = each bug report in bug_reports
# rawCorpus = raw data
# bug reports = dictionary of bug reports
# report time = fixedtime
# 2. n7ot elfixed time

# 3.ne7awel nlink 7etet elsrc file w nefhmha

# 3. netb3 elklam da f json file

def runBugHistory(bugReports):
    scores = []
    i=0
    global buggy_src_file
    for bugReport in bugReports.values():
        files = bugReport.fixedFiles;
        for buggy_src_file in files:
            buggy_src_file = os.path.normpath(buggy_src_file)

        mrReport = getMostRecentReport(buggy_src_file, convertToDateTime(bugReport.fixedTime),bugReports.values())
        bugFixingRecency_ = bugFixingRecency(bugReport, mrReport)
        i = i +1;
        print(i)
        scores.append(bugFixingRecency_)
    return scores


def getPreviousReportByFilename(filename, brdate, dictionary):
    return [br for br in dictionary if (filename in br.fixedFiles and convertToDateTime(br.fixedTime) < brdate)]


def convertToDateTime(date):
    return datetime.strptime(date, "%Y-%m-%d %H:%M:%S")


def getMonthsBetween(d1, d2):
    date1 = convertToDateTime(d1)
    date2 = convertToDateTime(d2)
    return abs((date1.year - date2.year) * 12 + date1.month - date2.month)


def getMostRecentReport(filename, currentDate, dictionary):
    matchingReports = getPreviousReportByFilename(filename, currentDate, dictionary)
    if len(matchingReports) > 0:
        return max((br for br in matchingReports), key=lambda x: convertToDateTime(x.fixedTime))
    else:
        return None


def bugFixingRecency(report1, report2):
    if report1 is None or report2 is None:
        return 0
    else:
        return 1 / float(getMonthsBetween(report1.fixedTime, report2.fixedTime) + 1)


def main():
    print("Bug history started")
    # Unpickle preprocessed data

    with open(swt.root + '/preprocessed_reports.pickle', 'rb') as file:
        bug_reports = pickle.load(file)
        runBugHistory(bug_reports)
    scores = runBugHistory(bug_reports)
    # Saving similarities in a json file
    with open(swt.root + '/bug_history.json', 'w') as file:
        json.dump(scores, file)
    print('Bug history finished')
    


if __name__ == '__main__':
    main()
