import re
from itertools import count
# DATA = "{1 4 5}, {1 2 4}, {4 5 7}, {1 2 5}, {4 5 8}, {1 5 9}, {1 3 6}, {2 3 4}, {5 6 7}, {3 4 5}, {3 5 6}, {3 5 7}, {6 8 9}, {3 6 7}, {3 6 8}"
DATA = "{1 2 3}, {1 4 5}, {1 2 4}, {1 2 5}, {1 5 9}, {1 3 6}, {2 3 4}, {2 5 9}, {3 4 5}, {3 5 6}, {3 5 9}, {3 8 9}, {3 2 6}, {4 5 7}, {4 1 8}, {4 7 8}, {4 6 7}, {6 1 3}, {6 3 4}, {6 8 9}, {6 2 1}, {6 4 3}, {6 7 9},{8 2 4}, {8 9 1}, {8 3 6}, {8 3 7}, {8 4 7}, {8 5 1}, {8 3 1}, {8 6 2}"

MAX_LEAF_SIZE = 3
ITEM_SET_LENGTH = 3
HASH_FUNCTION_VAL = 3

class LinkedListNode:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList:
    _ids = count(0)
    def __init__(self):
        self.name = str(next(self._ids))+"'"
        self.head = None
    
    def insert(self,data):
        if (self.head == None):
            self.head = LinkedListNode(data)
            return
        else:
            node = self.head
            while(node.next != None):
                node = node.next
            node.next = LinkedListNode(data)
    
    def getData(self):
        if (self.head == None):
            return None
        else:
            data = []
            node = self.head
            data.append(node.data)
            while(node.next != None):
                node = node.next
                data.append(node.data)
            return data
    
    def __repr__(self):
        return "(Linked List() - "+self.name+")"

    def __str__(self):
        string = self.name
        if (self.head == None):
            string =  "None"
        else:
            node = self.head
            string = string+": " + str(node.data)
            while(node.next != None):
                node = node.next
                string = string + "->"+ "{"+str(node.data)+ "}"
        return string
        


class Node:
    def __init__(self,name):
        ## For debug purpose
        self.name = name

        ## The data contains, when full (which isLeaf = False), no longer use
        self.data = []

        ## The data connected, in other words, its children, when isLeaf = False, will only use this
        self.connect = []

        for i in range(HASH_FUNCTION_VAL):
            self.connect.append(None)
        self.isLeaf = True
    
class HashTree:
    def __init__(self):
        self.root = Node("root")
        self.__linkedListPtr = []

    def insert(self, dataInsert):
        assert(len(dataInsert) == ITEM_SET_LENGTH)
        self.insertRecursion(self.root,dataInsert,1)
    
    def insertRecursion(self, node, dataInsert, layer):
        assert(len(dataInsert) == ITEM_SET_LENGTH)
        assert(isinstance(node,Node))

        if (node.isLeaf):
            ## Check if last layer of element and data in node full
            if (len(node.data) >= MAX_LEAF_SIZE and layer>ITEM_SET_LENGTH):

                ## Convert the last element to linked list, and link with new data
                # print("Need expand")
                # print("Before: "+str(node.data))
                if (not isinstance(node.data[MAX_LEAF_SIZE-1],LinkedList)):
                    tmp = node.data[MAX_LEAF_SIZE-1]
                    node.data[MAX_LEAF_SIZE-1] = LinkedList()
                    node.data[MAX_LEAF_SIZE-1].insert(tmp)
                # print("Append linked list")
                node.data[MAX_LEAF_SIZE-1].insert(dataInsert)
                # print("After: "+str(node.data))
                return
            
            ## If it is leaf, just append
            node.data.append(dataInsert)

            ## Check if the size of data in a node full
            if (len(node.data) > MAX_LEAF_SIZE):

                ## Perform splitting
                # print("---Splitting---")
                for value in node.data:
                    hashValue = (int(value[layer-1])+2)%HASH_FUNCTION_VAL
                    # print("Value Set "+str(value)+" Layer: "+str(layer)+" Value Compute: "+str(value[layer-1])+" Hash Value: "+str(hashValue))
                    if (node.connect[hashValue] == None):
                        node.connect[hashValue] = Node("Layer "+str(layer)+" with hash "+str(hashValue))
                    self.insertRecursion(node.connect[hashValue], value, layer+1)
                
                # print("After spit")
                # print("data: "+str(node.data))
                # print("connect: "+str(node.connect))
                # print(node.isLeaf)
                # print("---Finish Split---")
                
                del node.data
                ## Finish split
                node.isLeaf = False
        else:
            # print("Travse")
            # print(layer)

            ## Insert recursion, travse to the layer wanted if it is not a leaf node
            hashValue = (int(dataInsert[layer-1])+2)%HASH_FUNCTION_VAL
            if (node.connect[hashValue] == None):
                node.connect[hashValue]= Node("Layer "+str(layer)+" "+str(hashValue))
            
            self.insertRecursion(node.connect[hashValue], dataInsert, layer+1)
    
    def depthFirstSearch(self, node):
        if (node.isLeaf):
            # if (len(node.data) >= MAX_LEAF_SIZE and isinstance(node.data[MAX_LEAF_SIZE-1],list)):
            #     print("linked list result: ")
            #     print(node.data)
            for i in node.data:
                if (isinstance(i,LinkedList)):
                    self.__linkedListPtr.append(i)

            return node.data
        
        result = []
        for i in node.connect:
            if (isinstance(i,Node)):
                result.append(self.depthFirstSearch(i))
            else:
                result.append([])
        
        return result
    
    def getResult(self):
        self.__linkedListPtr = []
        return self.depthFirstSearch(self.root), self.__linkedListPtr


## Retrieve the input in within list bounded by '{' '}', with ' ' split
def processRawInput(inputValue):
    processed = inputValue.split(',')
    result = []
    for data in processed:
        data = re.search('{(.*)}', data)
        if (data):
            result.append(data.group(1).split(" "))
    return result

        

if __name__ == "__main__":
    hashTree = HashTree()
    datas = processRawInput(DATA)

    sortedDatas = []
    for i in datas:
        sortedDatas.append(tuple(sorted(i)))

    sortedDatas = list(dict.fromkeys(sortedDatas))

    count = 0
    for i in sortedDatas:
        # print("===============")
        # print(i)
        count = count + 1
        hashTree.insert(tuple(sorted(i)))
        
        # print("Intermediate Result: ")
        # print(hashTree.getResult())
        # if (count > 2):
        #     break
        # print("===============")
    print("Result: ")
    result = hashTree.getResult()
    if (len(result) > 1):
        print(result[0])
        print("Linked list details: ")
        for linkedList in result[1]:
            print(str(linkedList))

    else:
        print(result)