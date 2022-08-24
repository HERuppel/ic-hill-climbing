import os
from random import randint
import pandas as pd
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn import metrics

def loadData():
  dataframe = pd.read_csv(
    filepath_or_buffer=os.path.join(
      os.path.dirname(__file__), 'sets\dataR2.csv'),
      header=None,
      delimiter=","
  )

  data = dataframe.values

  return data[:, :-1], data[:, -1]

def hillClimbing(X_train, y_train, X_test, y_test, hiddenLayers):
  clf = MLPClassifier(max_iter=10000, alpha=1e-5, hidden_layer_sizes=hiddenLayers, random_state=5)
  clf.fit(X_train, y_train)
  sc = clf.score(X_test, y_test)
  print(sc)

  Ypred = clf.predict(X_test)
  accuracy = metrics.accuracy_score(y_test, Ypred)
  return clf.loss_curve_[::-1], round(accuracy * 100, 2)



X, y = loadData()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=5)

#Primeira camada 2 - 15
scores1, accuracy1 = hillClimbing(X_train, y_train, X_test, y_test, randint(2, 15))

#Segunda camada 0 - 10 *(1 - 10)
scores2, accuracy2 = hillClimbing(X_train, y_train, X_test, y_test, randint(1, 10))

print(accuracy1)
pyplot.plot(scores1)
pyplot.show()
print(accuracy2)
pyplot.plot(scores2)
pyplot.show()