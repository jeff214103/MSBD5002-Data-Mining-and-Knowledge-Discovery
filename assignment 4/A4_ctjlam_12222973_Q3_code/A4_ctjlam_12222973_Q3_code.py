import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DISTANCE_THRESHOLD = 10
MIN_PT = 2

def readFile(filename):
    data = []
    with open(filename,'r') as f:
        for line in f:
            x,y = map(float,line.split(' '))
            data.append([x,y])
    return np.array(data)

def nestedLoopDetection(data, distance_threshold, min_pt):
    outlier = np.array([0]*len(data))
    for i in range(len(data)):
        x = data.iloc[i]['x']
        y = data.iloc[i]['y']
        distance = data[(np.sqrt(np.square(x - data['x'])+np.square(y - data['y'])) <= distance_threshold) & (data.index != i)].index
        if (len(distance) <= min_pt):
            outlier[i] = 1
    return outlier



if __name__ == "__main__":
    data = readFile("Nested_Points.txt")
    dataframe = pd.DataFrame(data, columns=['x','y'])
    outlier = nestedLoopDetection(dataframe, DISTANCE_THRESHOLD, MIN_PT)
    dataframe['outlier'] = outlier
    for i in range(2):
        data = dataframe[dataframe['outlier'] == i]
        if i == 1:
            print("Distance Threshold: {} min pt: {} - Number of Outlier: {}".format(DISTANCE_THRESHOLD,MIN_PT,len(data)))
            print(data)
            plt.scatter(data['x'],data['y'], c="red", label=data['outlier'])
        else:
            plt.scatter(data['x'],data['y'], label=data['outlier'])
    plt.title("Nested Loop - Distance Threshold: {} min pt: {}".format(DISTANCE_THRESHOLD,MIN_PT))
    plt.savefig("nestedLoop.png")
    plt.show()