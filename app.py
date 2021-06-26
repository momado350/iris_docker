from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# load data
iris = load_iris()
x = iris.data
y = iris.target

# split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, random_seed=42, test_size=0.5)

# build the model
clf = RandomForestClassifier(n_estimators=10)

# train the classifier
clf.fit(x_train, y_train)

# make predictions
predicted = clf.predict(x_test)

# evaluate the model
print(accuracy_score(predicted, y_test))
                     