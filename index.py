import os
from random import randint
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.datasets import make_classification
from numpy.random import seed
from sklearn.model_selection import train_test_split

seed(1)

def loadData():
  dataframe = pd.read_csv(
    filepath_or_buffer=os.path.join(
      os.path.dirname(__file__), 'sets\semeion.data'),
      delimiter=r"\s+"
  )

  data = dataframe.values

  return data[:, :-1], data[:, -1]

def evaluate_predictions(y_test, yhat):
	return metrics.accuracy_score(y_test, yhat)

def random_predictions(n_examples):
	return [randint(0, 1) for _ in range(n_examples)]

def modify_predictions(current, n_changes=1):
	updated = current.copy()
	for i in range(n_changes):
		ix = randint(0, len(updated)-1)
		updated[ix] = 1 - updated[ix]
	return updated

def hillClimbing(X_test, y_test, maxIterations):
	scores = list()

	solution = random_predictions(X_test.shape[0])
	score = evaluate_predictions(y_test, solution)
	scores.append(score)

	for i in range(maxIterations):
		scores.append(score)
		if score == 1.0:
			break
		candidate = modify_predictions(solution)
		value = evaluate_predictions(y_test, candidate)
		if value >= score:
			solution, score = candidate, value
	return solution, scores

def classification(X_train, y_train, X_test, y_test, hidden_layer_sizes):
  clf = MLPClassifier(
    solver='lbfgs',
    max_iter=10000, 
    alpha=1e-5, 
    hidden_layer_sizes=hidden_layer_sizes, 
    random_state=1)
  clf.fit(X_train, y_train)

  Ypred = clf.predict(X_test)
  accuracy = metrics.accuracy_score(y_test, Ypred)
  return round(accuracy * 100, 2)



#X, y = loadData()
X, y = make_classification(n_samples=1000, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
test = hillClimbing(X_test, y_test, 1000)
score = classification(X_train, y_train, X_test, y_test, (randint(2, 15),randint(1, 10)))

print(score)
#Primeira camada 2 - 15
#scores1, accuracy1 = hillClimbing(X_train, y_train, X_test, y_test, (randint(2, 15),randint(1, 10)))

#Segunda camada 0 - 10 *(1 - 10)
#scores2, accuracy2 = hillClimbing(X_train, y_train, X_test, y_test, (randint(1, 10),))

#pyplot.plot(test[1])
#pyplot.show()
# print(accuracy2)
# pyplot.plot(scores2)
# pyplot.show()