import glob
import os.path
from collections import OrderedDict
import xmltodict
import javalang
import pygments
import pygments.lexers
from pygments.lexers.jvm import JavaLexer
from pygments.token import Token


# DataSet Class holds the information about eeach dataset paths
class DataSet:
    def __init__(self, name, srcCode, bugReport, root):
        self.name = name
        self.srcCode = srcCode
        self.bugReport = bugReport
        self.root = root


# Source codes and bug reports paths
aspectj = DataSet('aspectj', 'AspectJ/AspectJ-1.5', 'AspectJ/AspectJBugRepository.xml', 'AspectJ')
swt = DataSet('swt', 'SWT/SWT-3.1', 'SWT/SWTBugRepository.xml', 'SWT')
zxing = DataSet('zxing', 'ZXing/ZXing-1.6', 'ZXing/ZXingBugRepository.xml', 'ZXing')


# Parsing Data Sets

class BugReport:
    """Class representing each bug report"""

    def __init__(self, summary, description, fixedFiles, fixedTime,openDate):
        self.summary = summary
        self.description = description
        self.fixedFiles = fixedFiles
        self.fixedTime = fixedTime
        self.openDate = openDate
        self.pos_tagged_summary = None
        self.pos_tagged_description = None



class SourceCode:  # edited

    def __init__(self, fullCode, comments, classNames, attributes, methodNames, variables, fileName, packageName):
        self.fullCode = fullCode
        self.comments = comments
        self.classNames = classNames
        self.attributes = attributes
        self.methodNames = methodNames
        self.variables = variables
        self.fileName = fileName
        self.exact_file_name = fileName[0]
        self.packageName = packageName
        self.pos_tagged_comments = None

        self.srcFixedDate = None


class Parser:

    def __init__(self, project):
        self.name = project.name
        self.src = project.srcCode
        self.bugReport = project.bugReport

    # Parse the xml files (Bug Reports)
    def bugReportParser(self):

        # Convert XML bug repository to a dictionary
        with open(self.bugReport) as xml_file:
            xml_dict = xmltodict.parse(xml_file.read(), force_list={'file': True})

        # Iterate through bug reports and build their objects
        bug_reports = OrderedDict()

        for bug_report in xml_dict['bugrepository']['bug']:
            bug_reports[bug_report['@id']] = BugReport(bug_report['buginformation']['summary'],
                bug_report['buginformation']['description']
                if bug_report['buginformation']['description'] else '',
                [os.path.normpath(path) for path in bug_report['fixedFiles']['file']], bug_report['@fixdate'] if bug_report['@fixdate'] else '', bug_report['@opendate'] if bug_report['@opendate'] else ''

            )

        return bug_reports

    # Parse  (Source Code)
    def srcCodeParser(self):

        # Getting all the files with java extension recursively
        srcCodeAddresses = glob.glob(str(self.src) + '/**/*.java', recursive=True)  #

        # Creating a java lexer instance
        java_lexer = JavaLexer()
        src_files = OrderedDict()

        # iterating to parse each source file
        for src_file in srcCodeAddresses:
            with open(src_file) as file:
                src = file.read()

            # Attribute for each part of a source file
            comments = ''
            classNames = []
            attributes = []
            methodNames = []
            variables = []

            # Source parsing
            parseTree = None
            try:
                parseTree = javalang.parse.parse(src)
                for path, node in parseTree.filter(javalang.tree.VariableDeclarator):
                    # print(path)
                    if isinstance(path[-2], javalang.tree.FieldDeclaration):  # Item second to last
                        attributes.append(node.name)

                    elif isinstance(path[-2], javalang.tree.VariableDeclaration):
                        variables.append(node.name)
            except:
                pass

            # Trimming the source file
            ind = False
            if parseTree:
                if parseTree.imports:
                    last_imp_path = parseTree.imports[-1].path  # Last item
                    #print
                    src = src[src.index(last_imp_path) + len(last_imp_path) + 1:]

                elif parseTree.package:
                    package_name = parseTree.package.name
                    src = src[src.index(package_name) + len(package_name) + 1:]

                else:  # There is neither import nor package declaration
                    ind = True

            # no parse tree
            else:
                ind = True

            # Lexically tokenize the source file
            lexed_src = pygments.lex(src, java_lexer)

            for i, token in enumerate(lexed_src):
                # if its a comment add it at comments
                if token[0] in Token.Comment:
                    if ind and i == 0 and token[0] is Token.Comment.Multiline:
                        src = src[src.index(token[1]) + len(token[1]):]
                        continue
                    comments += token[1]

                # if its a class add it to class names
                elif token[0] is Token.Name.Class:
                    classNames.append(token[1])

                # if its a function ad it to method name
                elif token[0] is Token.Name.Function:
                    methodNames.append(token[1])

            # Get the package declaration
            if parseTree and parseTree.package:
                package_name = parseTree.package.name
            else:
                package_name = None

            # Handling special case for AspectJ dataset
            if self.name == 'aspectj':
                src_files[os.path.relpath(src_file, start=self.src)] = SourceCode(src, comments, classNames, attributes,methodNames, variables,[os.path.basename(src_file).split('.')[0]],package_name)
            else:
                # If source file has package declaration
                if package_name:
                    src_id = (package_name + '.' + os.path.basename(src_file))
                else:
                    src_id = os.path.basename(src_file)

                src_files[src_id] = SourceCode(src, comments, classNames, attributes, methodNames, variables, [os.path.basename(src_file).split('.')[0]], package_name)

        return src_files


def test():
    print('Dataset Test: ', zxing.name, zxing.srcCode, zxing.bugReport)
    parser = Parser(aspectj)
    x = parser.bugReportParser()
    y = parser.srcCodeParser()
    src_id, src = list(y.items())[10]
    print('Parsing Test: ', src_id, src.exact_file_name, src.packageName)


# Run test
if __name__ == '__main__':
    test()
