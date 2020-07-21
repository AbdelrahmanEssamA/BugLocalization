import string
import re
import nltk
import inflection
import pickle
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from Datasets import Parser, zxing, aspectj, swt


class ReportPreprocessing:

    def __init__(self, bugReports):

        self.bugReports = bugReports

    # part of speech tagging to identify its category
    def posTagging(self):

        for report in self.bugReports.values():
            summaryTokens = nltk.word_tokenize(report.summary)
            descriptionTokens = nltk.word_tokenize(report.description)
            posTaggedSummary = nltk.pos_tag(summaryTokens)
            posTaggedDescription = nltk.pos_tag(descriptionTokens)
            report.pos_tagged_summary = [token for token, pos in posTaggedSummary if 'NN' in pos or 'VB' in pos]
            report.pos_tagged_description = [token for token, pos in posTaggedDescription if 'NN' in pos or 'VB' in pos]

    # Splitting the bug report into tokens
    def tokenize(self):

        for report in self.bugReports.values():
            report.summary = nltk.wordpunct_tokenize(report.summary)
            report.description = nltk.wordpunct_tokenize(report.description)

    # Splitting camelcase to different words
    def splitCamelcase(self, tokens):

        # Copy tokens
        returningTokens = tokens[:]

        for token in tokens:
            splitTokens = re.split(fr'[{string.punctuation}]+', token)

            # If token is split into some other tokens
            if len(splitTokens) > 1:
                # Remove the old Camel case Token to be split
                returningTokens.remove(token)
                for st in splitTokens:
                    # Turning the camelcase into "_" then split them
                    camelCaseSplit = inflection.underscore(st).split('_')

                    if len(camelCaseSplit) > 1:
                        returningTokens.append(st)
                        returningTokens += camelCaseSplit

                    else:
                        returningTokens.append(st)

            else:
                camelCaseSplit = inflection.underscore(token).split('_')
                if len(camelCaseSplit) > 1:
                    returningTokens += camelCaseSplit

        return returningTokens

    # Removing punctuations and return the tokens in lower case
    def removePunctuation(self):

        # This var hod all the punctuations and numbers
        punctuationTable = str.maketrans({c: None for c in string.punctuation + string.digits})

        for report in self.bugReports.values():
            # remove the digits and punctuations using the punctuation table
            summaryNoPunc = [token.translate(punctuationTable) for token in report.summary]
            descriptionNoPunc = [token.translate(punctuationTable) for token in report.description]
            pos_sum_NoPunc = [token.translate(punctuationTable) for token in report.pos_tagged_summary]
            pos_desc_NoPunc = [token.translate(punctuationTable) for token in report.pos_tagged_description]

            # Transforming the upper case letters to lower case
            report.summary = [token.lower() for token in summaryNoPunc if token]
            report.description = [token.lower() for token in descriptionNoPunc if token]
            report.pos_tagged_summary = [token.lower() for token in pos_sum_NoPunc if token]
            report.pos_tagged_description = [token.lower() for token in pos_desc_NoPunc if token]

    # Removing stop word
    def removeStopwords(self):

        stopwords = nltk.corpus.stopwords.words('english')

        # Iterating and filtering the stop words from tokens
        for report in self.bugReports.values():
            report.summary = [token for token in report.summary if token not in stopwords]
            report.description = [token for token in report.description if token not in stopwords]
            report.pos_tagged_summary = [token for token in report.pos_tagged_summary if token not in stopwords]
            report.pos_tagged_description = [token for token in report.pos_tagged_description if token not in stopwords]

    # Removing java Keywords
    def removeJavaKeywords(self):

        # All the java keywords (   Reference Geeks for Geeks)
        java_keywords = {'abstract', 'assert', 'boolean', 'break', 'byte', 'case', 'catch', 'char', 'class', 'const',
                         'continue', 'default', 'do', 'double', 'else', 'enum', 'extends', 'final', 'finally',
                         'float', 'for', 'goto', 'if', 'implements', 'import', 'instanceof', 'int', 'interface', 'long',
                         'native', 'new', 'null', 'package', 'private', 'protected', 'public', 'return', 'short',
                         'static', 'strictfp', 'super', 'switch', 'synchronized', 'this', 'throw', 'throws',
                         'transient', 'try', 'void', 'volatile', 'while'}

        for report in self.bugReports.values():
            report.summary = [token for token in report.summary if token not in java_keywords]
            report.description = [token for token in report.description if token not in java_keywords]
            report.pos_tagged_summary = [token for token in report.pos_tagged_summary if token not in java_keywords]
            report.pos_tagged_description = [token for token in report.pos_tagged_description if token not in java_keywords]

    # stem the tokens ( return the words to its origin )
    def stem(self):

        # Stemmer instance (snow ball)
        sbs = SnowballStemmer("english")

        for report in self.bugReports.values():
            report.summary = dict(zip(['stemmed', 'unstemmed'], [[sbs.stem(token) for token in report.summary], report.summary]))
            report.description = dict(zip(['stemmed', 'unstemmed'], [[sbs.stem(token) for token in report.description], report.description]))
            report.pos_tagged_summary = dict(zip(['stemmed', 'unstemmed'], [[sbs.stem(token) for token in report.pos_tagged_summary], report.pos_tagged_summary]))
            report.pos_tagged_description = dict(zip(['stemmed', 'unstemmed'], [[sbs.stem(token) for token in report.pos_tagged_description], report.pos_tagged_description]))

    # Lemmatizing the tokens
    # def Lemmatize(self):
    # lemmatizer = WordNetLemmatizer()
    # Running preprocessing functions for bug reports
    def preprocess(self):

        self.posTagging()
        self.tokenize()
        # Running camelcase function for all report sections needed
        for report in self.bugReports.values():
            report.summary = self.splitCamelcase(report.summary)
            report.description = self.splitCamelcase(report.description)
            report.pos_tagged_summary = self.splitCamelcase(report.pos_tagged_summary)
            report.pos_tagged_description = self.splitCamelcase(report.pos_tagged_description)
        self.removePunctuation()
        self.removeStopwords()
        self.removeJavaKeywords()
        self.stem()


def main():
    # Parsing the data of the dataset to make it ready for preprocess
    parser = Parser(zxing)

    # Preprocess the data
    preprocessedReports = ReportPreprocessing(parser.bugReportParser())
    preprocessedReports.preprocess()

    # Creating a pickle file to hold the preprocessed data
    with open(zxing.root + '/preprocessed_reports.pickle', 'wb') as file:
        pickle.dump(preprocessedReports.bugReports, file, protocol=pickle.HIGHEST_PROTOCOL)

    print("Bug report preprocessed successfully")


if __name__ == '__main__':
    main()
