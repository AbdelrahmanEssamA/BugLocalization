import string
import re
import nltk
import inflection
import pickle
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from Datasets import  zxing, aspectj, swt
from Datasets import Parser


class SrcPreprocessing:

    def __init__(self, srcFiles):
        self.srcFiles = srcFiles

    # Part of speech tagging for the comments part only
    def posTagging(self):

        for src in self.srcFiles.values():
            # Tokenizing using word_tokeize instead of wordpunct_tokenize to be more accurate
            commentsTokens = nltk.word_tokenize(src.comments)
            posTaggedComments = nltk.pos_tag(commentsTokens)
            src.pos_tagged_comments = [token for token, pos in posTaggedComments if 'NN' in pos or 'VB' in pos]

    # Tokenizing the source code elements into Tokens
    def tokenize(self):

        for src in self.srcFiles.values():
            src.fullCode = nltk.wordpunct_tokenize(src.fullCode)
            src.comments = nltk.wordpunct_tokenize(src.comments)

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
                # Camel case detection for new tokens
                for st in splitTokens:
                    # Turning the camelcase into "_" then split them
                    camel_split = inflection.underscore(st).split('_')

                    if len(camel_split) > 1:
                        returningTokens.append(st)
                        returningTokens += camel_split

                    else:
                        returningTokens.append(st)

            else:
                camel_split = inflection.underscore(token).split('_')
                if len(camel_split) > 1:
                    returningTokens += camel_split

        return returningTokens

    # Removing punctuations and return the tokens in lower case
    def removePunctuation(self):

        # This var hod all the punctuations and numbers
        punctuationTable = str.maketrans({c: None for c in string.punctuation + string.digits})

        for src in self.srcFiles.values():
            # remove the digits and punctuations using the punctuation table
            fullCodeNoPunc = [token.translate(punctuationTable) for token in src.fullCode]
            commentsNoPunc = [token.translate(punctuationTable) for token in src.comments]
            classNamesNoPunc = [token.translate(punctuationTable) for token in src.classNames]
            attributesNoPunc = [token.translate(punctuationTable) for token in src.attributes]
            methodNamesNoPunc = [token.translate(punctuationTable) for token in src.methodNames]
            variablesNoPunc = [token.translate(punctuationTable) for token in src.variables]
            filenameNoPunc = [token.translate(punctuationTable) for token in src.fileName]
            posTaggedCommentsNoPunc = [token.translate(punctuationTable) for token in src.pos_tagged_comments]

            # Transforming the upper case letters to lower case
            src.fullCode = [token.lower() for token in fullCodeNoPunc if token]
            src.comments = [token.lower() for token in commentsNoPunc if token]
            src.classNames = [token.lower() for token in classNamesNoPunc if token]
            src.attributes = [token.lower() for token in attributesNoPunc if token]
            src.methodNames = [token.lower() for token in methodNamesNoPunc if token]
            src.variables = [token.lower() for token in variablesNoPunc if token]
            src.fileName = [token.lower() for token in filenameNoPunc if token]
            src.pos_tagged_comments = [token.lower() for token in posTaggedCommentsNoPunc if token]

    # Removing stop word
    def removeStopWords(self):

        stopwords = nltk.corpus.stopwords.words('english')

        # Iterating and filtering the stop words
        for src in self.srcFiles.values():
            src.fullCode = [token for token in src.fullCode if token not in stopwords]
            src.comments = [token for token in src.comments if token not in stopwords]
            src.classNames = [token for token in src.classNames if token not in stopwords]
            src.attributes = [token for token in src.attributes if token not in stopwords]
            src.methodNames = [token for token in src.methodNames if token not in stopwords]
            src.variables = [token for token in src.variables if token not in stopwords]
            src.fileName = [token for token in src.fileName if token not in stopwords]
            src.pos_tagged_comments = [token for token in src.pos_tagged_comments if token not in stopwords]

    # Removing java Keywords
    def removeJavaKeywords(self):

        # Java language keywords
        javaKeywords = {'abstract', 'assert', 'boolean', 'break', 'byte', 'case', 'catch', 'char', 'class', 'const',
                        'continue', 'default', 'do', 'double', 'else', 'enum', 'extends', 'false', 'final', 'finally',
                        'float', 'for', 'goto', 'if', 'implements', 'import', 'instanceof', 'int', 'interface', 'long',
                        'native', 'new', 'null', 'package', 'private', 'protected', 'public', 'return', 'short',
                        'static', 'strictfp', 'super', 'switch', 'synchronized', 'this', 'throw', 'throws',
                        'transient', 'true', 'try', 'void', 'volatile', 'while'}

        for src in self.srcFiles.values():
            src.fullCode = [token for token in src.fullCode if token not in javaKeywords]
            src.comments = [token for token in src.comments if token not in javaKeywords]
            src.classNames = [token for token in src.classNames if token not in javaKeywords]
            src.attributes = [token for token in src.attributes if token not in javaKeywords]
            src.methodNames = [token for token in src.methodNames if token not in javaKeywords]
            src.variables = [token for token in src.variables if token not in javaKeywords]
            src.fileName = [token for token in src.fileName if token not in javaKeywords]
            src.pos_tagged_comments = [token for token in src.pos_tagged_comments if token not in javaKeywords]

    # Stemming tokens
    def stem(self):
        # Stemmer instance (snow ball)
        stemmer = SnowballStemmer("english")

        for src in self.srcFiles.values():

            src.fullCode = dict(zip(['stemmed', 'unstemmed'], [[stemmer.stem(token) for token in src.fullCode], src.fullCode]))
            src.comments = dict(zip(['stemmed', 'unstemmed'],[[stemmer.stem(token) for token in src.comments], src.comments]))
            src.classNames = dict(zip(['stemmed', 'unstemmed'], [[stemmer.stem(token) for token in src.classNames], src.classNames]))
            src.attributes = dict(zip(['stemmed', 'unstemmed'], [[stemmer.stem(token) for token in src.attributes], src.attributes]))
            src.methodNames = dict(zip(['stemmed', 'unstemmed'], [[stemmer.stem(token) for token in src.methodNames], src.methodNames]))
            src.variables = dict(zip(['stemmed', 'unstemmed'], [[stemmer.stem(token) for token in src.variables], src.variables]))
            src.fileName = dict(zip(['stemmed', 'unstemmed'], [[stemmer.stem(token) for token in src.fileName], src.fileName]))
            src.pos_tagged_comments = dict(zip(['stemmed', 'unstemmed'], [[stemmer.stem(token) for token in src.pos_tagged_comments], src.pos_tagged_comments]))


    # Running preprocessing functions for src code files
    def preprocess(self):

        self.posTagging()
        self.tokenize()
        # Running camelcase function for all src files sections needed
        for src in self.srcFiles.values():
            src.fullCode = self.splitCamelcase(src.fullCode)
            src.comments = self.splitCamelcase(src.comments)
            src.classNames = self.splitCamelcase(src.classNames)
            src.attributes = self.splitCamelcase(src.attributes)
            src.methodNames = self.splitCamelcase(src.methodNames)
            src.variables = self.splitCamelcase(src.variables)
            src.fileName = self.splitCamelcase(src.fileName)
            src.pos_tagged_comments = self.splitCamelcase(src.pos_tagged_comments)
        self.removePunctuation()
        self.removeStopWords()
        self.removeJavaKeywords()
        self.stem()


def main():
    # Parsing the data of the dataset to make it ready for preprocess
    parser = Parser(zxing)
    # Preprocess the data
    print("Src Code preprocessing started")
    preprocessedSrcFiles = SrcPreprocessing(parser.srcCodeParser())
    preprocessedSrcFiles.preprocess()

    # Creating a pickle file to hold the preprocessed data
    with open(zxing.root + '/preprocessed_src.pickle', 'wb') as file:
        pickle.dump(preprocessedSrcFiles.srcFiles, file, protocol=pickle.HIGHEST_PROTOCOL)

    print("Src Code preprocessed successfully")


if __name__ == '__main__':
    main()
