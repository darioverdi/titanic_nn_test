import pandas
import numpy as np
import neural_network
import time


begin = time.time()
csv_reader = pandas.read_csv("train.csv", usecols=[1, 2, 4, 5, 6, 7, 9, 11])
headers = list(csv_reader.columns.values)
num_features = 7
num_examples = len(getattr(csv_reader, headers[0]))
X = []
y = []

# Fill in missing data with the median or the most common value
csv_reader["Age"] = csv_reader["Age"].fillna(csv_reader["Age"].median())
csv_reader["Embarked"] = csv_reader["Embarked"].fillna("S")

# Clean data
csv_reader.loc[csv_reader["Survived"] == 0, "Survived"] = 2  # Replace 0 with 2 in the survived column
csv_reader.loc[csv_reader["Sex"] == "male", "Sex"] = 1       # Replace male with 1 in the sex column
csv_reader.loc[csv_reader["Sex"] == "female", "Sex"] = 2     # Replace female with 2 in the sex column
csv_reader.loc[csv_reader["Embarked"] == "S", "Embarked"] = 1  # Replace S with 1 in the embarked column
csv_reader.loc[csv_reader["Embarked"] == "C", "Embarked"] = 2  # Replace C with 2 in the embarked column
csv_reader.loc[csv_reader["Embarked"] == "Q", "Embarked"] = 3  # Replace Q with 3 in the embarked column
print "Data cleaned successfully"

# Collect y
y = np.array(csv_reader["Survived"])
print "y collected successfully"

# Collect X
for i in range(num_examples):
    X.append(map(float, csv_reader[headers].iloc[i, 1:]))
X = np.array(X)
print "X collected successfully"

# Processing the test set
csv_test = pandas.read_csv("test.csv", usecols=[0, 1, 3, 4, 5, 6, 8, 10])
headers_test = list(csv_test.columns.values)
n = 8
m = len(getattr(csv_test, headers_test[0]))
X_test = []
csv_test["Age"] = csv_test["Age"].fillna(csv_reader["Age"].median())
csv_test["Fare"] = csv_test["Fare"].fillna(csv_test["Fare"].median())
csv_test.loc[csv_test["Sex"] == "male", "Sex"] = 1
csv_test.loc[csv_test["Sex"] == "female", "Sex"] = 2
csv_test["Embarked"] = csv_test["Embarked"].fillna("S")
csv_test.loc[csv_test["Embarked"] == "S", "Embarked"]= 1
csv_test.loc[csv_test["Embarked"] == "C", "Embarked"] = 2
csv_test.loc[csv_test["Embarked"] == "Q", "Embarked"] = 3
print "Test set processed successfully"

# Collect X_test
for i in range(m):
    X_test.append(map(float, csv_test[headers_test].iloc[i, 1:]))
X_test = np.array(X_test)
print "X test collected successfully"

# Train nn
net = neural_network.NeuralNetwork(X, y, 2, num_features, num_examples, 25, 0.3)
time0 = time.time()
net.train(100)
print "nn trained successfully in %f" % (time.time()-time0)

# Evaluate
accuracy = 0
for i in range(num_examples):
    out = net.predict(X[i])
    if y[i] == net.interpret_result(out):
        accuracy += 1
print "Accuracy is %f%%" % (float(accuracy)/num_examples*100)
print "Evaluation done"

# Perform on test set
y_test = []
for i in range(m):
    out = net.predict(X_test[i])
    pred = net.interpret_result(out)
    if pred == 2:
        y_test.append(0)
    else:
        y_test.append(pred)
t0 = time.time()
print "Prediction done successfully in %f" % (time.time()-t0)

# Write to CSV
final_result = pandas.DataFrame({
        "PassengerId": csv_test["PassengerId"],
        "Survived": y_test
    })
final_result.to_csv("kaggle.csv", index=False)
print "Done everything in %f" % (time.time()-begin)