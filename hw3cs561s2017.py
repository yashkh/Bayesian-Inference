
import sys
import copy
from decimal import Decimal
import itertools
#-----------------------------------Function and Definitions------------------------------

#Splits the given literal into its variable and the value(eg. LeakIdea = +)
def splitLiteral(literal):
    literal = literal.strip()
    holder = literal.split(' = ')
    variable = holder[0].strip()
    value = holder[1].strip()
    value = True if value == '+' else False
    return variable,value

#Sorts the nodes in the topological order and returns the topological sorted list of nodes
def topologicalSort(bayesnet):
    nodes = bayesnet.keys()
    sortedNodes = []

    while len(sortedNodes) < len(nodes):
        for node in nodes:
            if node not in sortedNodes and all(parent in sortedNodes for parent in bayesnet[node]['parents']):
                sortedNodes.append(node)

    return sortedNodes

#Returns only the node that are required for the query.
def nodeSelection(evidence,bayesnet,sortedNodes):
    addedNodeSet = set(evidence.keys())
    isNodePresent = [True if x in addedNodeSet else False for x in sortedNodes]

    while len(addedNodeSet) != 0:
        popNode = addedNodeSet.pop()
        for parent in bayesnet[popNode]['parents']:
            addedNodeSet.add(parent)
            parentIndex = sortedNodes.index(parent)
            isNodePresent[parentIndex] = True

    newSortedNodes = []
    for node in sortedNodes:
        if isNodePresent[sortedNodes.index(node)] == True:
            newSortedNodes.append(node)

    return newSortedNodes

#Returns the probability using enumeration given the required variables(vars),evidence variables(e) and the bayesian network.
def enumeration(vars,e,bayesnet):
    if len(vars) == 0:
        return 1.0

    Y = vars[0]

    if Y in e:
        result = probability(Y,e,bayesnet)*enumeration(vars[1:],e,bayesnet)
    else:
        sumProbability = []
        e2 = copy.deepcopy(e)
        for y in [True,False]:
            e2[Y] = y
            sumProbability.append(probability(Y,e2,bayesnet)*enumeration(vars[1:],e2,bayesnet))
        result =sum(sumProbability)

    return result

#Returns the probability of variable Y given its parents in evidence e.
def probability(Y,e,bayesnet):

    if bayesnet[Y]['type'] == 'decision':
        return 1.0

    if len(bayesnet[Y]['parents']) == 0:
        if e[Y] == True:
            return float(bayesnet[Y]['prob'])
        else:
            return 1.0-float(bayesnet[Y]['prob'])
    else:
        parentTuple = tuple(e[parent] for parent in bayesnet[Y]['parents'])

        if e[Y] == True:
            return float(bayesnet[Y]['condprob'][parentTuple])
        else:
            return 1-float(bayesnet[Y]['condprob'][parentTuple])


#-----------------------------------Input & Building Data Structures--------------------------------

#Bayesian Network Dictionary
BayesNet = {}
sortedNodes = []
rawQueryList = []
#Reading the input file

#filename = sys.argv[-1]
filename = 'input02.txt'
inputFile = open(filename)

#Building queries from input
line = inputFile.readline().strip()

while line != '******':
    rawQueryList.append(line)
    #print line
    line = inputFile.readline().strip()


#Building the bayesian network from input
line = ' '
while line != '':
    #Declaring the parent list
    parents = []
    # Input the node names and the parents
    line = inputFile.readline().strip()

    nodeAndParents = lines = line.split(' | ')
    node = nodeAndParents[0].strip()

    if len(nodeAndParents) != 1:
        parents = nodeAndParents[1].strip().split(' ')

    BayesNet[node] = {}
    BayesNet[node]['parents'] = parents
    BayesNet[node]['children']=[]

    #Insert child for all the parents
    for parent in parents:
        BayesNet[parent]['children'].append(node)

    # Input the probabilities

    if len(parents) == 0:
        line = inputFile.readline().strip()
        if line == 'decision':
            #Decision Node
            BayesNet[node]['type'] = 'decision'
        else:
            #Node with prior probability
            BayesNet[node]['type'] = 'normal'
            BayesNet[node]['prob'] = line
    else:
        #Nodes with conditional probabilies
        condprob = {}
        for i in range(0,pow(2,len(parents))):
            line = inputFile.readline().strip()
            lines = line.split(' ')
            prob = lines[0]
            lines = lines[1:]
            truth = tuple(True if x == '+' else False for x in lines)
            condprob[truth] = prob

        BayesNet[node]['type'] = 'normal'
        BayesNet[node]['condprob'] = condprob

    line = inputFile.readline().strip()
#print BayesNet['E']
#-------------------------------Declaring Output Files----------------------------------------
outputFile = open('output.txt','w')

#print BayesNet
#--------------------------------Query Inferencing---------------------------------------------

#Sort all the nodes in topological order
sortedNodes = topologicalSort(BayesNet)

#Query Inferencing for all the input queries
for query in rawQueryList:

    fullEvidence = {}
    observedEvidence = {}

    operation = query[:query.index('(')]
    operation = operation.strip()

    if operation == 'P':
        #print 'Operation P'
        isSeparatorGiven = False
        result = 1.0
        #print query
        literals = query[query.index('(')+1:query.index(')')]
        orIndex = literals.index('|') if '|' in literals else -1
        #print orIndex
        #If both query and evidence is given.
        if orIndex != -1:
            isSeparatorGiven = True
            holder = literals[:orIndex]
            #print holder
            xLiterals = holder.strip()
            xLiterals = xLiterals.split(',')
            for xLiteral in xLiterals:
                xLiteral = xLiteral.strip()
                xVar,xVal = splitLiteral(xLiteral)
                fullEvidence[xVar] = xVal

            holder = literals[orIndex+1:]
            #print holder.strip()
        #If only evidence is given
        else:
            holder = literals

        literals = holder.strip()
        literals = literals.split(',')
        for literal in literals:
            literal = literal.strip()
            var,val = splitLiteral(literal)
            fullEvidence[var] = val
            observedEvidence[var] = val

        #Final calculations
        if isSeparatorGiven == True:
            #Calculating the numerator
            sortedNodesForNumerator = nodeSelection(fullEvidence,BayesNet,sortedNodes)
            numerator = enumeration(sortedNodesForNumerator,fullEvidence,BayesNet)

            #Calculating the denominator
            sortedNodesForDenominator = nodeSelection(observedEvidence,BayesNet,sortedNodes)
            denominator = enumeration(sortedNodesForDenominator,observedEvidence,BayesNet)

            result = numerator/denominator

        else:
            sortedNodesForQuery = nodeSelection(observedEvidence,BayesNet,sortedNodes)
            result = enumeration(sortedNodesForQuery,observedEvidence,BayesNet)

        #print 'Full Evidence:',fullEvidence
        #print 'Observed Evidence:',observedEvidence

        result = Decimal(str(result+1e-8)).quantize(Decimal('.01'))
        #print result
        outputFile.write(str(result))
        outputFile.write('\n')


    elif operation == 'EU':
        #print 'Operation EU'
        isSeparatorGiven = False
        result = 1.0

        literals = query[query.index('(')+1:query.index(')')]
        orIndex = literals.index('|') if '|' in literals else -1

        #If both query and evidence is given.
        if orIndex != -1:
            isSeparatorGiven = True
            holder = literals[:literals.index(' | ')]

            xLiterals = holder.strip()
            xLiterals = xLiterals.split(',')
            for xLiteral in xLiterals:
                xLiteral = xLiteral.strip()
                xVar,xVal = splitLiteral(xLiteral)
                fullEvidence[xVar] = xVal

            holder = literals[literals.index(' | ')+3:]
        #If only evidence is given
        else:
            holder = literals

        literals = holder.strip()
        literals = literals.split(',')
        for literal in literals:
            literal = literal.strip()
            var,val = splitLiteral(literal)
            fullEvidence[var] = val
            observedEvidence[var] = val

        fullEvidence['utility'] = True

        #Final calculations
        if isSeparatorGiven == True:
            #Calculating the numerator
            sortedNodesForNumerator = nodeSelection(fullEvidence,BayesNet,sortedNodes)
            numerator = enumeration(sortedNodesForNumerator,fullEvidence,BayesNet)

            #Calculating the denominator
            sortedNodesForDenominator = nodeSelection(observedEvidence,BayesNet,sortedNodes)
            denominator = enumeration(sortedNodesForDenominator,observedEvidence,BayesNet)

            result = numerator/denominator

        else:
            sortedNodesForQuery = nodeSelection(fullEvidence,BayesNet,sortedNodes)
            result = enumeration(sortedNodesForQuery,fullEvidence,BayesNet)

        #print 'Full Evidence:',fullEvidence
        #print 'Observed Evidence:',observedEvidence

        result = int(round(result))
        #print result
        outputFile.write(str(result))
        outputFile.write('\n')


    else:
        #print 'Operation MEU'

        isSeparatorGiven = False
        result = {}
        maximizationLiterals = []

        literals = query[query.index('(')+1:query.index(')')]
        orIndex = literals.index('|') if '|' in literals else -1

        #If both query and evidence is given.
        if orIndex != -1:
            isSeparatorGiven = True
            holder = literals[:literals.index(' | ')]

            xLiterals = holder.strip()
            xLiterals = xLiterals.split(',')
            for xLiteral in xLiterals:
                equalIndex = xLiteral.index('=') if '=' in xLiteral else -1
                if equalIndex != -1:
                    xLiteral = xLiteral.strip()
                    xVar,xVal = splitLiteral(xLiteral)
                    fullEvidence[xVar] = xVal
                else:
                    maximizationLiterals.append(xLiteral.strip())

            holder = literals[literals.index(' | ')+3:]
        #If only evidence is given
        else:
            holder = literals

        literals = holder.strip()
        literals = literals.split(',')
        for literal in literals:
            equalIndex = literal.index('=') if '=' in literal else -1
            if equalIndex != -1:
                literal = literal.strip()
                var,val = splitLiteral(literal)
                fullEvidence[var] = val
                observedEvidence[var] = val
            else:
                maximizationLiterals.append(literal.strip())

        fullEvidence['utility'] = True

        print "maximize", maximizationLiterals
        sizeOfMaxLiterals = len(maximizationLiterals)
        truthTable = list(itertools.product([True, False], repeat=sizeOfMaxLiterals))

        for i in range(0,len(truthTable)):
            completeEvidence = copy.deepcopy(fullEvidence)
            value = ''
            j = 0
            for maxLiteral in maximizationLiterals:
                completeEvidence[maxLiteral] = truthTable[i][j]
                if truthTable[i][j] == True:
                    value = value + '+ '
                else:
                    value = value + '- '
                j = j+1



            #Final calculations
            if isSeparatorGiven == True:
                #Calculating the numerator
                sortedNodesForNumerator = nodeSelection(completeEvidence,BayesNet,sortedNodes)
                numerator = enumeration(sortedNodesForNumerator,completeEvidence,BayesNet)

                #Calculating the denominator
                sortedNodesForDenominator = nodeSelection(observedEvidence,BayesNet,sortedNodes)
                denominator = enumeration(sortedNodesForDenominator,observedEvidence,BayesNet)

                eachResult = numerator/denominator

            else:
                sortedNodesForQuery = nodeSelection(completeEvidence,BayesNet,sortedNodes)
                eachResult = enumeration(sortedNodesForQuery,completeEvidence,BayesNet)

            result[eachResult] = value

        maxResult = max(result.keys())

        #print result[maxResult]+str(int(round(maxResult)))
        outputFile.write(result[maxResult]+str(int(round(maxResult))))
        outputFile.write('\n')

#-----------------------------Brushing the final output file--------------------------------
outputFile.close()
with open('output.txt', 'rb+') as finalStripFile:
    finalStripFile.seek(0,2)
    size=finalStripFile.tell()
    finalStripFile.truncate(size-1)
    finalStripFile.close()


