import numpy as np

X = [0,1,2,3,4,5,6,7,8,9]
Y = [1,1,-1,-1,-1,1,1,-1,-1,1]

### Code if library is not allowed

def getSeperator():
    # Get all seperator for continous value
    seperator = []
    for i in range(len(X)-1):
        seperator.append((X[i]+X[i+1])/2)
    return seperator

## For reference calculation
## Calculating Gini Impurity
def calGiniImpurity(dictionary):
    sum = 0
    for key in dictionary:
        sum = sum + dictionary[key]

    if (sum == 0):
        sum = 1

    giniImpurity = 1
    for key in dictionary:
        giniImpurity = giniImpurity - (dictionary[key]/sum)*(dictionary[key]/sum)
    return giniImpurity

## For Handling conitinous value
## Find a seperator which best sep the continous dataset
def determineContinousValueCutOff(X,Y,sample_weights=np.ones(len(Y))):
    weights = np.array(sample_weights)
    
    # Get all seperator for continous value
    seperator = getSeperator()
    
    suggestedSeperator = 0
    minGini = 1
    labelForSmallThreshold = None
    labelForBigThreshold = None

    # Store all the sperator gini impurity
    for sep in seperator:
        smallCount = 0
        smallCountResult = {1:0, -1:0}

        bigCount = 0
        bigCountResult = {1:0, -1:0}

        #Loop through array for counting the seperator and its label
        for i in range(len(X)):
            if (X[i]<sep):
                smallCount = smallCount + float(weights[i])
                smallCountResult[Y[i]] = smallCountResult[Y[i]] + float(weights[i])
            else:
                bigCount = bigCount + float(weights[i])
                bigCountResult[Y[i]] = bigCountResult[Y[i]] + float(weights[i])

        smallCountGiniImpurity = calGiniImpurity(smallCountResult)
        bigCountGiniImpurity = calGiniImpurity(bigCountResult)
        
        sum = smallCount+bigCount
        giniImpurity = (smallCount/sum)*smallCountGiniImpurity+(bigCount/sum)*bigCountGiniImpurity
        if (giniImpurity < minGini):

            # Store the best result for separotor
            suggestedSeperator = sep
            minGini = giniImpurity
            labelForSmallThreshold = max(smallCountResult, key=smallCountResult.get)
            labelForBigThreshold = max(bigCountResult, key=bigCountResult.get)

            
    return suggestedSeperator, labelForSmallThreshold,labelForBigThreshold


## Used to compute the accuaacy of the model
def calAccuracy(Y_actual, Y_predict, weights=np.ones(len(Y))):
    assert(isinstance(Y_actual, list) or isinstance(Y_predict,np.ndarray))
    assert(isinstance(Y_predict, list) or isinstance(Y_predict,np.ndarray))
    assert(isinstance(weights, list) or isinstance(weights,np.ndarray))
    assert(len(Y_actual) == len(Y_predict))

    arraySize = len(Y_actual)
    accuracy = 0
    for i in range(arraySize):
        if (Y_actual[i] == Y_predict[i]):
            accuracy = accuracy+weights[i]
    return accuracy/arraySize


## a really weak classifer for classify the model (x>v) then large threshold, else small threshold
class WeakClassifier:
    def __init__(self):
        self.threshold = None
        self.smallLabel = None
        self.largeLabel = None

    ## Use to train and get the threshold and the labels
    def fit(self, X, Y,sample_weights=np.ones(len(Y))):
        threshold, labelForSmallThreshold, labelForLargeThreshold = determineContinousValueCutOff(X,Y,sample_weights)
        self.threshold = threshold
        self.smallLabel = labelForSmallThreshold
        self.largeLabel = labelForLargeThreshold

    ## Used to predict an unknown X
    def predict(self, X):
        assert(isinstance(X, list) or isinstance(X,np.ndarray))
        predicted = []
        for x in X:
            if x>=self.threshold:
                predicted.append(self.largeLabel)
            else:
                predicted.append(self.smallLabel)
        return np.array(predicted)
    
    def __str__(self):
        return "Weak Classifier - x > v (where v = "+str(self.threshold)+") then y = "+str(self.largeLabel)+" else y = "+str(self.smallLabel)
### End of Code if library is not allowed

class AdaBoost:
    def __init__(self, numberOfClassifier):
        self.alphas = []
        self.classifier = []
        self.numberOfClassifier = numberOfClassifier
        assert len(self.classifier) == len(self.alphas)


    def fit(self, X, Y):
        # Clear before calling
        self.alphas = []
        weights = np.ones(len(Y)) / len(Y)
        # Iterate over numberOfClassifier weak classifiers
        while len(self.alphas) < self.numberOfClassifier:

            #Random sample X, Y
            sample_index = np.random.choice(range(len(Y)), size=len(Y), replace=True, p=weights)

            # #Get random sample X training and Y training
            x_train = np.array(X)[sample_index.astype(int)]
            y_train = np.array(Y)[sample_index.astype(int)]
            weights_train = (np.ones(len(Y)) / len(Y))

            ## Without random sample and without fructuration
            # x_train = X
            # y_train = Y
            # weights_train = weights


            #For library usage code
            # weakClassifier = DecisionTreeClassifier(max_depth = 1)
            # weakClassifier.fit(X, Y, sample_weight = weights)

            # Self defined weak classifier
            weakClassifier = WeakClassifier()

            # Train the weak classifier for the threshold and labels
            weakClassifier.fit(x_train,y_train,weights_train)

            ## Do prediction for the weak classifier
            y_pred = weakClassifier.predict(X)
                  
            #Not equal = 1, equal = 0
            compare = (np.not_equal(Y, y_pred)).astype(int)

            #Only consider the weights not equal
            error = (sum(weights * compare))/len(Y)

            # If error > 0.5, then just go back to first step
            if (error > 0.5):
                continue

            # Append with the weak classifier
            self.classifier.append(weakClassifier) 

            # Update the importance of this classifier
            alpha = (1/2)*np.log((1 - error) / error)
            self.alphas.append(alpha)

            #Not equal = 1, equal = -1
            compare[compare==0] = -1

            #Weights update
            weights = weights * np.exp(alpha * compare)

            #Weights normalization
            weights = weights/sum(weights)
            
    def predict(self, X):
        scores = np.zeros(len(Y))
        # Predict class label for each weak classifier, weighted by alpha_m
        for i in range(self.numberOfClassifier):
            scores += self.classifier[i].predict(X) * self.alphas[i]

        return np.sign(scores)

    def printAdaBoostInfo(self):

        for i in range(self.numberOfClassifier):
            print("========== Classifier "+str(i+1) +" ==========")
            print("Aplhas: "+str(self.alphas[i]))

            # For library version
            # tree_info = tree.export_text(self.classifier[i])
            # print("Tree: ")
            # print(tree_info)

            print(self.classifier[i])


            print("======================================")
            


if __name__ == "__main__":
    X = np.array(X).reshape(-1,1)
    ada = AdaBoost(5)
    ada.fit(X,Y)
    y_pred = ada.predict(X)
    ada.printAdaBoostInfo()
    print("Prediction: "+str(y_pred))
    print("Accuracy: "+str(calAccuracy(Y,y_pred)))