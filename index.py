import os
from random import randint
import pandas as pd
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.datasets import make_classification
from numpy.random import seed

seed(1)

def loadData():
  dataframe = pd.read_csv(
    filepath_or_buffer=os.path.join(
      os.path.dirname(__file__), 'sets\semeion.data'),
      delimiter=r"\s+"
  )

  data = dataframe.values

  return data[:, :-1], data[:, -1]

def hillClimbing(X_train, y_train, X_test, y_test, hidden_layer_sizes):
  clf = MLPClassifier(
    early_stopping=True, 
    max_iter=10000, 
    alpha=1e-5, 
    hidden_layer_sizes=hidden_layer_sizes, 
    tol=1e-8, 
    random_state=1,
    learning_rate_init=.01)
  clf.fit(X_train, y_train)

  Ypred = clf.predict(X_test)
  accuracy = metrics.accuracy_score(y_test, Ypred)
  #[::-1] reverse
  print(clf.validation_scores_)
  return clf.validation_scores_, round(accuracy * 100, 2)



#X, y = loadData()
X, y = make_classification(n_samples=1000, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

#Primeira camada 2 - 15
scores1, accuracy1 = hillClimbing(X_train, y_train, X_test, y_test, (randint(2, 15),randint(1, 10)))

#Segunda camada 0 - 10 *(1 - 10)
#scores2, accuracy2 = hillClimbing(X_train, y_train, X_test, y_test, (randint(1, 10),))

print(accuracy1)
pyplot.plot(scores1)
pyplot.show()
# print(accuracy2)
# pyplot.plot(scores2)
# pyplot.show()