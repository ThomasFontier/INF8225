from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def loadDataset(fraudProportion: float = 0.5, testProportion: float = 0.25):
    # Preliminar args checkings 
    assert fraudProportion > 0 and fraudProportion <= 1
    assert testProportion > 0 and testProportion <= 0.5

    # Reading CSV data file
    df = pd.read_csv(r"S-FFSD.csv")
    df["Target"] = df["Target"].str[1:].astype(int)
    df["Source"] = df["Source"].str[1:].astype(int)
    df["Location"] = df["Location"].str[1:].astype(int)
    df["Type"] = df["Type"].str[2:].astype(int)
    
    # Take only labels 1 or 0
    fraudArray = df.loc[df["Labels"] == 1]
    nonFraudArray = df.loc[df["Labels"] == 0]
    fraudArray = fraudArray.drop("Labels", axis=1).to_numpy()
    nonFraudArray = nonFraudArray.drop("Labels", axis=1).to_numpy()
    
    # Randomize elements order
    np.random.shuffle(fraudArray)
    np.random.shuffle(nonFraudArray)
    
    # Training/Test Splitting
    nTestFraudArray = int(fraudArray.shape[0]*testProportion)
    nTestNonFraudArray = int(nonFraudArray.shape[0]*testProportion)
    
    testFraudArray = fraudArray[:nTestFraudArray]
    testNonFraudArray = nonFraudArray[:nTestNonFraudArray]
    
    trainFraudArray = fraudArray[nTestFraudArray:]
    trainNonFraudArray = nonFraudArray[nTestNonFraudArray:]
    
    # Setting Fraud Proportion
    dataSize = int(trainFraudArray.shape[0]/fraudProportion) + 1
    nNonFraud = dataSize - trainFraudArray.shape[0]
    trainNonFraudArray = nonFraudArray[:nNonFraud]
    
    return trainNonFraudArray, trainFraudArray, testNonFraudArray, testFraudArray

if __name__ == "__main__":
    
    accuracy = 0
    nRuns = 5
    fraudProportion = 0.5
    
    for i in range(nRuns):
        trainNonFraudArray, trainFraudArray, testNonFraudArray, testFraudArray = loadDataset(fraudProportion)
        
        trainDatata = np.vstack([trainNonFraudArray, trainFraudArray])
        gmm = GaussianMixture(n_components=2)
        gmm.fit(trainDatata)
        
        labels = np.concatenate((np.full((testNonFraudArray.shape[0], 2), np.asarray([0, 1])),np.full((testFraudArray.shape[0], 2), np.asarray([1, 0]))))
        testData = np.concatenate((testNonFraudArray, testFraudArray))
        probabilities = gmm.predict_proba(testData)
        predictions = np.zeros_like(probabilities)
        predictions[np.arange(len(probabilities)), probabilities.argmax(1)] = 1
        accuracy += np.sum(np.where(predictions == labels, 0.5, 0)) / labels.shape[0]
    print("TEST ACCURACY: ", accuracy/nRuns)