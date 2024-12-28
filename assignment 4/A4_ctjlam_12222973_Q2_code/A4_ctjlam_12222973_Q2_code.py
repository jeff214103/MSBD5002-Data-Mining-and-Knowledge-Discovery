import numpy as np
import random
import pandas as pd
from enum import Enum
import matplotlib.pyplot as plt
from queue import Queue

PARAMETER_SETTING = [(3,5),(2.5,20),(2,20),(2,15), (1.3,8)]

def readFile(filename):
    data = []
    with open(filename,'r') as f:
        for line in f:
            x,y = map(float,line.split(' '))
            data.append([x,y])
    return np.array(data)

class PtRelationship(Enum):
    CORE = 1
    BORDER = 2
    NOISE = 3

class DBScan:
    def __init__(self, epsilon, min_pts):
        self.epsilon = epsilon
        self.min_pts = min_pts

    def __findNeighbours(self, data, index):
        x = data.iloc[index]['x']
        y = data.iloc[index]['y']

        #find all datapts inside the circle
        neighbours = data[(np.sqrt(np.square(x - data['x'])+np.square(y - data['y'])) <= self.epsilon) & (data.index != index)].index

        return neighbours, self.__getRelation(neighbours)

    def __getRelation(self, data):
        if (len(data) >= self.min_pts):
            return PtRelationship.CORE
        elif (len(data) < self.min_pts and len(data) > 0):
            return PtRelationship.BORDER
        elif (len(data) == 0):
            return PtRelationship.NOISE

    def fit(self, data):
        n_cluster = 1

        cluster = np.array([0]*len(data)) #Init all point as 0, which is noisy point
        
        unvisited = list(range(len(data)))

        while(len(unvisited) > 0):

            index = random.choice(unvisited)
            unvisited.remove(index)

            neighbours_index, relation = self.__findNeighbours(data,index)

            if (relation == PtRelationship.CORE):
                cluster[index] = n_cluster

                queue = Queue()
                for idx in neighbours_index:
                    queue.put(idx)

                while(queue.empty() == False):
                    neighbour = queue.get()
                    cluster[neighbour] = n_cluster
                    if (neighbour in unvisited):
                        neighbours_neighbour_index, relation = self.__findNeighbours(data,neighbour)
                        unvisited.remove(neighbour)
                        if (relation == PtRelationship.CORE):
                            for idx in neighbours_neighbour_index:
                                cluster[idx] = n_cluster
                                if (idx in unvisited):
                                    queue.put(idx)
                
                n_cluster = n_cluster+1
        
        return cluster, n_cluster


if __name__ == '__main__':
    original_data = readFile("DBSCAN_Points.txt")
    dataframe = pd.DataFrame(original_data, columns=['x','y'])

    for epsilon, min_pt in PARAMETER_SETTING:
        dbScan = DBScan(epsilon=epsilon,min_pts=min_pt)
        cluster, numberOfCluster = dbScan.fit(dataframe)
        # print("ep: {} min_pt: {} - Number of cluster: {}".format(epsilon,min_pt,numberOfCluster))
        dataframe['cluster'] = cluster
        
        for i in range(numberOfCluster+1):
            data = dataframe[dataframe['cluster'] == i]
            if i == 0:
                print("ep: {} min_pt: {} - Number of Outlier: {}".format(epsilon,min_pt,len(data)))
                plt.scatter(data['x'],data['y'], c="black", label=data['cluster'])
            else:
                plt.scatter(data['x'],data['y'], label=data['cluster'])
        plt.title("DB Scan (ep: {} min_pt: {})".format(epsilon,min_pt))
        plt.savefig("DBScan_ep_{}_minpt_{}.png".format(epsilon,min_pt))
        plt.show()