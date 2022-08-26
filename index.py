import os
import time
import warnings
from random import randint
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
import pandas as pd
from matplotlib import pyplot
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

def loadData():
  dataframe = pd.read_csv(
    filepath_or_buffer=os.path.join(
      os.path.dirname(__file__), 'sets\dataR2.csv'),
      header=None,
      delimiter=","
  )

  data = dataframe.values

  return data[:, :-1], data[:, -1]

def evalPrediction(Y, candidate):
  return metrics.accuracy_score(Y, candidate)

def randPredictions(numExamples):
  return [randint(0, 1) for _ in range(numExamples)]

def modifyPredictions(currentSolution, numChanges=1):
  updated = currentSolution.copy()
  for _ in range(numChanges):
    ix = randint(0, len(updated) - 1)
    updated[ix] = 1 - updated[ix]
  return updated

def classification_hillclimbing(X_train, X_test, y_train, y_test):
    bestAccuracy = -np.inf
    bestI = -1
    bestJ = -1
    scores = list()
    solution = randPredictions(X.shape[0])
    score = evalPrediction(Y, solution)
    specificities = list()
    sensitivities = list()
    trainScores = list()
    testScores = list()

    for i in range(2, 15): 
      print("Testando {}".format("." * i))
      for j in range(0, 10):
        hiddenLayerSizes = (i) if j < 2 else (i, j)
        clf = MLPClassifier(solver='lbfgs', max_iter=30000, alpha=1e-5, hidden_layer_sizes=hiddenLayerSizes, random_state=int(time.time()))
        clf.fit(X_train, y_train)

        yPred = clf.predict(X_test)
        
        tn, fp, fn, tp = confusion_matrix(y_test, yPred).ravel()
        specificity = tn / (tn+fp)
        specificities.append(specificity)

        sensitivity = tp / (tp + fn)
        sensitivities.append(sensitivity)

        trainPred = clf.predict(X_train)
        trainAcc = metrics.accuracy_score(y_train, trainPred)
        trainScores.append(trainAcc)

        testPred = clf.predict(X_test)
        testAcc = metrics.accuracy_score(y_test, testPred)
        testScores.append(testAcc)

        candidate = modifyPredictions(solution)
        value = evalPrediction(Y, candidate)
        if value >= score:
          solution, score = candidate, value
          bestAccuracy = value
          bestI = i
          bestJ = j
        scores.append(score)
    return bestAccuracy, bestI, bestJ, scores, specificities, sensitivities, trainScores, testScores


X, Y = loadData()
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=int(time.time()))
bestAccuracy, bestI, bestJ, scores, specificities, sensitivities, trainScores, testScores = classification_hillclimbing(X_train, X_test, y_train, y_test)

print("Melhor acurácia: {}%".format(round(bestAccuracy * 100, 2)))
print("Nós do melhor resultado - 1ª Camada: {} | 2ª Camada: {} ".format(bestI, bestJ))


pyplot.figure(num='Acurácia')
pyplot.title('Acurácia a cada iteração')
pyplot.xlabel('Iterações')
pyplot.ylabel('Acurácia')
pyplot.plot(scores)
pyplot.show()

pyplot.figure(num='Especificidade')
pyplot.title('Especificidade a cada iteração')
pyplot.xlabel('Iterações')
pyplot.ylabel('Especificidade')
pyplot.plot(specificities)
pyplot.show()

pyplot.figure(num='Sensibilidade')
pyplot.title('Sensibilidade a cada iteração')
pyplot.xlabel('Iterações')
pyplot.ylabel('Sensiblididade')
pyplot.plot(sensitivities)
pyplot.show()

pyplot.figure(num='Overfitting')
pyplot.plot(trainScores, label='Treino')
pyplot.plot(testScores, label='Teste')
pyplot.legend()
pyplot.show()