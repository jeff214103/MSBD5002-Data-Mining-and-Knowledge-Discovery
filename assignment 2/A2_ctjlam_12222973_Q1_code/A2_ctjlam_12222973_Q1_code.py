def getData():
    import pandas as pd
    PATH = "./dataset/"

    TRAIN_PATH = PATH+"winequality_train.csv"
    TEST_PATH = PATH+"winequality_test.csv"

    train_data = pd.read_csv(TRAIN_PATH, sep=';')
    test_data = pd.read_csv(TEST_PATH, sep=';')

    #Extract features for training data
    X_train = train_data.iloc[:,:-1]
    #Extract labels for training data
    Y_train = train_data.iloc[:,-1:]

    #Extract features for testing data
    X_test = test_data.iloc[:,:-1]
    #Extract labels for testing data
    Y_test = test_data.iloc[:,-1:]

    return X_train,Y_train, X_test,Y_test

def decisionTree(method, treeDepth, X_train, Y_train, X_test, Y_test):
    from sklearn.tree import DecisionTreeClassifier
    import time

    decisionTree = DecisionTreeClassifier(criterion=method, max_depth=treeDepth)
    start = time.time()
    decisionTree = decisionTree.fit(X_train,Y_train)
    stop = time.time()
    Y_pred = decisionTree.predict(X_test)

    return Y_pred, (stop-start)


def KNN(numberOfNeighbours,X_train, Y_train, X_test, Y_test):
    from sklearn.neighbors import KNeighborsClassifier
    import time
    import numpy as np
    Y_train = np.ravel(Y_train)
    Y_test = np.ravel(Y_test)

    
    knn = KNeighborsClassifier(n_neighbors=numberOfNeighbours)
    start = time.time()
    decisionTree = knn.fit(X_train,Y_train)
    stop = time.time()
    Y_pred = decisionTree.predict(X_test)

    return Y_pred, (stop-start)

def RandomForest(numberOfEstimator, X_train, Y_train, X_test, Y_test):
    from sklearn.ensemble import RandomForestClassifier
    import time
    import numpy as np
    Y_train = np.ravel(Y_train)
    Y_test = np.ravel(Y_test)

    randomForest = RandomForestClassifier(n_estimators=numberOfEstimator)
    start = time.time()
    decisionTree = randomForest.fit(X_train,Y_train)
    stop = time.time()
    Y_pred = decisionTree.predict(X_test)

    return Y_pred, (stop-start)

def report(Y_test,predict,time):
    from sklearn.metrics import classification_report
    print(f"Training time: {time}s")
    print(classification_report(Y_test, predict,zero_division=0))

if __name__=="__main__":
    X_train,Y_train, X_test,Y_test = getData()

    #Remove Header
    X_train.columns = range(X_train.shape[1])
    Y_train.columns = range(Y_train.shape[1])
    X_test.columns = range(X_test.shape[1])
    Y_test.columns = range(Y_test.shape[1])

    print("================================================")
    print("Summary for decision tree (Entropy, depth = 5)")
    predict, timeCost = decisionTree("entropy",5,X_train,Y_train, X_test,Y_test)
    report(Y_test,predict,timeCost)
    print("================================================")
    input("Press enter to continue for next...")
    print("================================================")
    print("Summary for decision tree (Entropy, depth = 10)")
    predict, timeCost = decisionTree("entropy",10,X_train,Y_train, X_test,Y_test)
    report(Y_test,predict,timeCost)
    print("================================================")
    input("Press enter to continue for next...")
    print("================================================")
    print("Summary for decision tree (Entropy, depth = 15)")
    predict, timeCost = decisionTree("entropy",15,X_train,Y_train, X_test,Y_test)
    report(Y_test,predict,timeCost)
    print("================================================")
    input("Press enter to continue for next...")
    print("================================================")
    print("Summary for decision tree (Entropy, depth = 20)")
    predict, timeCost = decisionTree("entropy",20,X_train,Y_train, X_test,Y_test)
    report(Y_test,predict,timeCost)
    print("================================================")
    input("Press enter to continue for next...")
    print("================================================")
    print("Summary for decision tree (gini, depth = 5)")
    predict, timeCost = decisionTree("gini",5,X_train,Y_train, X_test,Y_test)
    report(Y_test,predict,timeCost)
    print("================================================")
    input("Press enter to continue for next...")
    print("================================================")
    print("Summary for decision tree (gini, depth = 10)")
    predict, timeCost = decisionTree("gini",10,X_train,Y_train, X_test,Y_test)
    report(Y_test,predict,timeCost)
    print("================================================")
    input("Press enter to continue for next...")
    print("================================================")
    print("Summary for decision tree (gini, depth = 15)")
    predict, timeCost = decisionTree("gini",15,X_train,Y_train, X_test,Y_test)
    report(Y_test,predict,timeCost)
    print("================================================")
    input("Press enter to continue for next...")
    print("================================================")
    print("Summary for decision tree (gini, depth = 20)")
    predict, timeCost = decisionTree("gini",20,X_train,Y_train, X_test,Y_test)
    report(Y_test,predict,timeCost)
    print("================================================")
    input("Press enter to continue for next...")
    print("================================================")
    print("Summary for KNN (# Neighbours=3)")
    predict, timeCost = KNN(3, X_train,Y_train, X_test,Y_test)
    report(Y_test,predict,timeCost)
    print("================================================")
    input("Press enter to continue for next...")
    print("================================================")
    print("Summary for KNN (# Neighbours=5)")
    predict, timeCost = KNN(5, X_train,Y_train, X_test,Y_test)
    report(Y_test,predict,timeCost)
    print("================================================")
    input("Press enter to continue for next...")
    print("================================================")
    print("Summary for KNN (# Neighbours=7)")
    predict, timeCost = KNN(7, X_train,Y_train, X_test,Y_test)
    report(Y_test,predict,timeCost)
    print("================================================")
    input("Press enter to continue for next...")
    print("================================================")
    print("Summary for Raindom Forest (# Estimator=10)")
    predict, timeCost = RandomForest(10, X_train,Y_train, X_test,Y_test)
    report(Y_test,predict,timeCost)
    print("================================================")
    input("Press enter to continue for next...")
    print("================================================")
    print("Summary for Raindom Forest (# Estimator=50)")
    predict, timeCost = RandomForest(50, X_train,Y_train, X_test,Y_test)
    report(Y_test,predict,timeCost)
    print("================================================")
    input("Press enter to continue for next...")
    print("================================================")
    print("Summary for Raindom Forest (# Estimator=100)")
    predict, timeCost = RandomForest(100, X_train,Y_train, X_test,Y_test)
    report(Y_test,predict,timeCost)
    print("================================================")
    input("Press enter to exit...")
    print("================================================")