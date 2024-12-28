import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def readFile(filename):
    data = []
    with open(filename,'r') as f:
        for line in f:
            x,y,label = map(float,line.split(' '))
            data.append([x,y,label])
    return np.array(data)

class FuzzyClusteringEM:
    def __init__(self, n_cluster, iteration):
        self.__n_cluster = n_cluster
        self.__iteration = iteration
        self.__centers = [None]*self.__n_cluster

    def fit(self, fit_data):
        matrix = np.zeros((self.__n_cluster,fit_data.shape[0]))

        #Random init centers
        for i in range(self.__n_cluster):
            self.__centers[i] = fit_data[i][:2]
        
        self.__centers = np.array(self.__centers)

        #Loop for each iteration
        for i in range(self.__iteration):
            print("====== {} iteration ======".format(i+1))

            #Init distance for each cluster
            distance_per_clust = [None]*self.__n_cluster

            #Cal the distance from pt the cluster centers
            for j in range(self.__n_cluster):
                distance_per_clust[j] = np.sum(np.square(fit_data[:,:2] -  self.__centers[j]),axis=1)

            distance_per_clust = np.array(distance_per_clust)

            inverse_distance_per_clust = 1/(distance_per_clust+1e-32) #prevent zero division

            distance_sum = np.sum(inverse_distance_per_clust,axis=0)

            matrix = np.square(inverse_distance_per_clust/distance_sum)

            new_center = matrix.dot(fit_data[:,:2])/np.sum(matrix,axis=1, keepdims=True)

            print("Updated center: \n {}".format(new_center))

            SSE = (matrix*distance_per_clust).sum()

            print("SSE: {}".format(SSE))

            self.__centers = new_center
        
        result = np.argmax(matrix.transpose(), axis=1)
        return result


if __name__ == '__main__':
    data = readFile("EM_Points.txt")

    #Original Data
    dataframe = pd.DataFrame(data, columns=['x','y','label'])

    cluster = FuzzyClusteringEM(n_cluster=3, iteration=15)

    #Predicted Data
    predicted = cluster.fit(data)
    dataframe['predicted'] = predicted

    fig, ax = plt.subplots(2)

    ax[0].set_title('Original Label')
    for i in range(3):
        data = dataframe[dataframe['label'] == i]
        ax[0].scatter(data['x'],data['y'], label=data['label'])

    ax[1].set_title('Predicted Label (Iteration: {})'.format(15))
    for i in range(3):
        data = dataframe[dataframe['predicted'] == i]
        ax[1].scatter(data['x'],data['y'], label=data['predicted'])

    plt.tight_layout()
    plt.savefig("fuzzyCluster.png")
    plt.show()