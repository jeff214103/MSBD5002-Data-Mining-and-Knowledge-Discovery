from typing import Dict
import itertools
import sys
THRESHOLD = 2500
FILE_NAME = "DataSetA.csv"

# THRESHOLD = 3
# FILE_NAME = "exampleLecture.csv"

##Extracted from https://aclanthology.org/Y13-1045.pdf
# THRESHOLD = 3
# FILE_NAME = "newExample.csv"

#Q2 datset
# THRESHOLD = 2
# FILE_NAME = "q2dataSet.csv"

FREQUEN_ITEM_THRESHOLD = 0


## Step 1: Deduce the ordered frequent items. For items with
## the same frequency, the order is given by the alphabetical
## order
def step1(filename, threshold):
    frequencyTable = getFrequentAndOrderElement(buildFrequencyTable(filename),threshold)
    return frequencyTable, createOrderedFrequentItemset(filename, frequencyTable)

def buildFrequencyTable(filename):
    frequencyTable = {}
    with open(filename) as file:
        for line in file:
            itemset = line.rstrip()
            for item in itemset.split(","):
                if (item == ''):
                    continue
                if (item in frequencyTable.keys()):
                    frequencyTable[item] = frequencyTable[item]+1
                else:
                    frequencyTable[item] = 1
    return frequencyTable


def getFrequentAndOrderElement(frequencyTable, threshold):
    assert(isinstance(frequencyTable,dict))
    newFrequencyTable = {}
    for key in frequencyTable:
        if (frequencyTable[key] > threshold):
            newFrequencyTable[key] = frequencyTable[key]

    #First sort by Name
    sortByName = dict(sorted(newFrequencyTable.items(), key=lambda d: d[0]))
    #Sort by value
    return dict(sorted(sortByName.items(), key=lambda x: x[1], reverse=True))

def createOrderedFrequentItemset(filename, frequencyTable):
    orderedFrequentItems = []
    with open(filename) as file:
        count = 0
        for line in file:
            count = count + 1
            tmpFrequenctItems = []
            itemset = line.rstrip()
            
            for item in itemset.split(","):
                if (item == ''):
                    continue
                if item in frequencyTable.keys():
                    tmpFrequenctItems.append(item)
            if (tmpFrequenctItems != []):
                # print("Dict: "+str(frequencyTable))
                # print("Unordered List: "+str(tmpFrequenctItems))
                # print("Ordered: "+str(sorted(tmpFrequenctItems, key=lambda x: frequencyTable[x], reverse=True)))
                tmpFrequenctItems.sort()
                tmpFrequenctItems = sorted(tmpFrequenctItems, key=lambda x: frequencyTable[x], reverse=True)
                orderedFrequentItems.append(tmpFrequenctItems)
            
            # if (count > 24):
            #     break
    return orderedFrequentItems
## Step 1 end

## Step 2: Construct the FP-tree from the data
class FPNode:
    def __init__(self, transaction, value, parentNode):
        self.name = transaction
        self.value = value
        self.count = 1
        self.connected = []
        self.parent = parentNode
        self.neighbour = None
    
    def __str__(self, level=0):
        ret = "\t"*level+"("+repr(self.value)+","+repr(self.count)+")\n"
        for child in self.connected:
            ret += child.__str__(level+1)
        return ret

    def __repr__(self):
        return '<FPNode node representation>'

class FPTree:
    def __init__(self):
        self.root = FPNode("root", None, None)
        self.builtItems = {}
        self.__oldNodeRef = None
    
    def construct(self, frequentItemData):
        for i in range(len(frequentItemData)):
            transaction = frequentItemData[i]
            # print("Build Items: "+str(self.builtItems))
            # print("Transaction Hanlde: "+str(transaction))
            #Perform insert on each transaction starting from root 
            node = self.root
            if (isinstance(transaction, dict)):
                for data in transaction: 
                    index = next((i for i, item in enumerate(node.connected) if item.value == data), -1)
                    if (index == -1):
                        # print(str(i)+"th transaction - Create Node: "+str(data))
                        newNode = FPNode(str(data)+" - "+str(transaction), data, node)
                        newNode.count = transaction[data]
                        node.connected.append(newNode)
                        node = newNode
                    else:
                        node.connected[index].count = node.connected[index].count + transaction[data]
                        node = node.connected[index]
            else:
                for data in transaction:
                    index = next((i for i, item in enumerate(node.connected) if item.value == data), -1)
                    if (index == -1):
                        # print(str(i)+"th transaction - Create Node: "+str(data))
                        newNode = FPNode(str(data)+" - "+str(transaction), data, node)
                        node.connected.append(newNode)
                        node = newNode
                    else:
                        node.connected[index].count = node.connected[index].count + 1
                        node = node.connected[index]

        self.connectTree()
    
    def connectTree(self):
        self.__oldNodeRef = {}
        self.buildConnection(self.root)
        del self.__oldNodeRef

    def buildConnection(self, node):
        assert(isinstance(node,FPNode))

        for child in node.connected:
            if (child.value != None):
                if (child.value not in self.builtItems.keys()):
                    self.builtItems[child.value] = child
                    self.__oldNodeRef[child.value] = child
                else:
                    self.__oldNodeRef[child.value].neighbour = child
                    self.__oldNodeRef[child.value] = child
            self.buildConnection(child)

    ## Step 3##
    def constructCondPatternBase(self, originalData):
        conditionalPatternBases = []
        dataFrequencyTable = {}

        #Travsing the same item node
        horizontalNode = self.builtItems[originalData]
        while (True):
            tmpConditionalPattern = {}

            # Travse upward
            upwardNode = horizontalNode.parent
            while (upwardNode.parent != None):
                if (upwardNode.value not in dataFrequencyTable):
                    dataFrequencyTable[upwardNode.value] = 0
                dataFrequencyTable[upwardNode.value] = dataFrequencyTable[upwardNode.value] + horizontalNode.count
                tmpConditionalPattern[upwardNode.value] = horizontalNode.count
                upwardNode = upwardNode.parent
            
            conditionalPatternBases.append(tmpConditionalPattern)
            if (horizontalNode.neighbour != None):
                horizontalNode = horizontalNode.neighbour
            else:
                break
        
        ## Filter out unfrequent items
        processFrequencyTable = {}
        for key in dataFrequencyTable:
            if (dataFrequencyTable[key] >= THRESHOLD):
                processFrequencyTable[key] = dataFrequencyTable[key]

        ## Filter out pattern base without frequent item 
        filteredConditionalPatternBase = []
        for condionalPatternBase in conditionalPatternBases:
            tmpConditionalPattern = {}
            for key in condionalPatternBase:
                if (key in processFrequencyTable):
                    tmpConditionalPattern[key] = condionalPatternBase[key]
            filteredConditionalPatternBase.append(tmpConditionalPattern)


        return processFrequencyTable, filteredConditionalPatternBase  
    ## Step 3 end##

    ## Step 4 v2 ##
    def getFrequentDataPattern(self, frequencyTable, originalData, originalCount):
        patterns = []
        for key in self.builtItems:
            tmpFrequentPattern = {}
            if (frequencyTable[key] < THRESHOLD):
                continue
            else:
                tmpFrequentPattern[key] = frequencyTable[key]
            patterns.append(tmpFrequentPattern)
            horizontalNode = self.builtItems[key]
            while (True):
                tmpFrequentPattern = {}
                # Travse upward
                if (horizontalNode.count >= THRESHOLD):
                    tmpFrequentPattern[horizontalNode.value] = horizontalNode.count
                    upwardNode = horizontalNode.parent
                    while (upwardNode.parent != None):
                        tmpFrequentPattern[upwardNode.value] = horizontalNode.count
                        upwardNode = upwardNode.parent
                    if (len(tmpFrequentPattern)>1):
                        patterns.append(tmpFrequentPattern)
                
                if (horizontalNode.neighbour != None):
                    horizontalNode = horizontalNode.neighbour
                else:
                    break
        frequentDataPattern = {}
        frequentDataPattern[originalData] = originalCount
        for key in patterns:
            tmp = list(key).copy()
            if (tmp == None):
                continue
            frequentDataPattern[tuple([originalData]+tmp)] = key[min(key.keys(), key=(lambda k: key[k]))]
        return frequentDataPattern
    ## Step 4 v2 end##

    def printTree(self):
        print("root", end='')
        self.printTreeRecursion(self.root)
    
    def printTreeRecursion(self, node):
        assert(isinstance(node, FPNode))
        for child in node.connected:
            print(" => ",end='')
            print("("+str(child.value)+","+str(child.count)+")",end='')
            if (len(child.connected) == 0):
                print()
                return
            self.printTreeRecursion(child)
    
    def printRoot(self):
        print(self.root)
## Step 2 end
## Step 4 v1 ##
def getFrequentItemset(originalData, originalCount, conditionalFPTree, conditionalPatternBase):
    frequentItemSet = {}

    ## Noted that this line of code should be uncommented if we follow the rule by following links
    ## https://aclanthology.org/Y13-1045.pdf
    ## Else if follow lecture, then it should produce the result without single length dataset
    #######################################################
    frequentItemSet[originalData] = originalCount
    #######################################################

    for L in range(1, len(conditionalFPTree)+1):
        combinations = list(itertools.combinations(conditionalFPTree, L))
        for i in combinations:   
            combination = list(i)

            minValue = getCombinationFrequency(combination,conditionalPatternBase)
            if (minValue<THRESHOLD):
                continue

            combination.insert(0, originalData)
            key = tuple(combination)
            if key not in frequentItemSet:
                frequentItemSet[key] = minValue
            else:
                if minValue > frequentItemSet[key]:
                    frequentItemSet[key] = minValue
    return frequentItemSet

def getCombinationFrequency(combination, conditionalPatternBases):
    count = 0
    combination = set(combination)
    for conditionalPatternBase in conditionalPatternBases:
        if (combination.issubset(set(conditionalPatternBase.keys()))):
            count = count + next(iter(conditionalPatternBase.values()))

    return count
        

# ## Step 4 v1 end ##

if __name__ == "__main__":

    ## Step 1
    frequencyTable, frequentItemData = step1(FILE_NAME, FREQUEN_ITEM_THRESHOLD)

    ## Step 2 - Constructing FP Tree
    fpTree = FPTree()
    fpTree.construct(frequentItemData)

    # fpTree.printRoot()

    # print("Frequency Table: "+str(frequencyTable))

    result = {}
    ##Loop through frequent item only
    for targetData in frequencyTable:
        ## Skip unfrequent item
        if (frequencyTable[targetData] < THRESHOLD):
            continue
        
        # print("============================================")
        # print("Getting "+targetData+" Frequent Pattern...")

        ## Step 3 - Constructing FP conditional Tree
        dataFrequencyTable, conditionalPatternBases = fpTree.constructCondPatternBase(targetData)


        ## Other method
        ## Step 4 - Getting Frequent Itemset v2
        ## First sort by Name
        # dataFrequencyTable = dict(sorted(dataFrequencyTable.items(), key=lambda d: d[0]))
        # ## Sort by value
        # dataFrequencyTable = dict(sorted(dataFrequencyTable.items(), key=lambda x: x[1]))
        
        # orderedContinonPatternBases = []
        # for conditionPatternBase in conditionalPatternBases:
        #     if (len(conditionPatternBase) > 0):
        #         sortedConditionPatternBase =  dict(sorted(conditionPatternBase.items(), key=lambda d: d[0]))
        #         sortedConditionPatternBase =  dict(sorted(conditionPatternBase.items(), key=lambda d: dataFrequencyTable[d[0]]))
        #         orderedContinonPatternBases.append(sortedConditionPatternBase)
        # # print("============================================")
        # # print(dataFrequencyTable)
        # # print(orderedContinonPatternBases)
        # # print("Original Data: "+str(targetData))
        # conditionFPTree = FPTree()
        # conditionFPTree.construct(orderedContinonPatternBases)
        # frequentItemset = conditionFPTree.getFrequentDataPattern(dataFrequencyTable, targetData, frequencyTable[targetData])

        # # print(frequentItemset)
        
        # # print("============================================")
        # # print("Conditional FP Tree: "+str(dataFrequencyTable))
        # # print("Conditional Pattern Base: "+str(conditionalPatternBases))
        ## Step 4 - Getting Frequent Itemset v2 end

        ## Step 4 - Getting Frequent Itemset v1
        frequentItemset = getFrequentItemset(targetData,frequencyTable[targetData],dataFrequencyTable, conditionalPatternBases)
        ## Step 4 - Getting Frequent Itemset v1
        
        for key in frequentItemset:
            result[key] = frequentItemset[key]
        #     print("("+str(key)+") : "+str(frequentItemset[key]))
        # print("============================================")

    print("Final result: ")
    for key in dict(sorted(result.items(), key=lambda item: item[1], reverse=True)):
        print(str(key)+" : "+str(result[key]))

    
        