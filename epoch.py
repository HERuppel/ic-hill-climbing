import os
import time
import warnings
from random import randint
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
import pandas as pd
from matplotlib import pyplot

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

def classification_hillclimbing(X, Y):
    bestAccuracy = -np.inf
    bestI = -1
    bestJ = -1
    scores = list()
    solution = randPredictions(X.shape[0])
    score = evalPrediction(Y, solution)

    for i in range(2, 15): 
      print("Testando {}".format("." * i))
      for j in range(0, 10):
        hiddenLayerSizes = (i) if j < 2 else (i, j)
        clf = MLPClassifier(solver='lbfgs', max_iter=30000, alpha=1e-5, hidden_layer_sizes=hiddenLayerSizes, random_state=int(time.time()))
        clf.fit(X, Y)

        candidate = modifyPredictions(solution)
        value = evalPrediction(Y, candidate)
        if value >= score:
          solution, score = candidate, value
          bestAccuracy = value
          bestI = i
          bestJ = j
        scores.append(score)
    return bestAccuracy, bestI, bestJ, scores


X, Y = loadData()
bestAccuracy, bestI, bestJ, scores = classification_hillclimbing(X, Y)

print("Melhor acurácia: {}%".format(round(bestAccuracy * 100, 2)))
print("Nodos do melhor resultado - 1ª Camada: {} | 2ª Camada: {} ".format(bestI, bestJ))
pyplot.title('Acurácia a cada iteração')
pyplot.xlabel('Iterações')
pyplot.ylabel('Acurácia')
pyplot.plot(scores)
pyplot.show()