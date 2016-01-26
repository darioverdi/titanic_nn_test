import pandas
import numpy as np
import neural_network
import time
import re
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
import matplotlib.pyplot as plt


def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ""


def svm(X, y, X_test):
    param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5], 'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
    clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
    clf = clf.fit(X, y)
    y_pred = clf.predict(X_test)
    y_test = []
    for y in y_pred:
        if y == 2:
            y_test.append(0)
        else:
            y_test.append(1)
    return y_test


def collect_family_id(name, familysize):
    family_name = name.split(",")[0]
    id = family_name + str(familysize)
    global CURRENT_ID
    if id not in FAMILY_IDs:
        WOMEN_IN_FAMILY[CURRENT_ID] = 0
        FAMILY_IDs[id] = CURRENT_ID
        CURRENT_ID += 1


def count_women_in_family(csv):
    for i in range(len(csv)):
        if csv["Sex"][i] == "female":
            WOMEN_IN_FAMILY[csv["FamilyID"][i]] += 1


CURRENT_ID = 1
FAMILY_IDs = {}
WOMEN_IN_FAMILY = {}
begin = time.time()
csv_reader = pandas.read_csv("train.csv", usecols=[1, 2, 3, 4, 5, 6, 7, 9, 11])
headers = list(csv_reader.columns.values)
X = []
y = []

# Generate more features
csv_reader["Family"] = csv_reader["SibSp"] + csv_reader["Parch"]
csv_reader["NameLen"] = csv_reader["Name"].apply(lambda x: len(x))
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2, "Dona": 10}
csv_reader["Title"] = csv_reader["Name"].apply(lambda x: title_mapping[get_title(x)])
for i in range(len(csv_reader["Name"])):
    collect_family_id(csv_reader["Name"][i], csv_reader["Family"][i])
csv_reader["FamilyID"] = map(lambda x: x.split(",")[0], csv_reader["Name"])
IDs = []
for i in range((len(csv_reader["Name"]))):
    IDs.append(FAMILY_IDs[csv_reader["FamilyID"][i]+str(csv_reader["Family"][i])])
csv_reader["FamilyID"] = IDs
count_women_in_family(csv_reader)
csv_reader["WomenFamily"] = csv_reader["FamilyID"].apply(lambda x: WOMEN_IN_FAMILY[x])

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

# # Select features
# predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked",
#               "Family", "Title", "FamilyID", "WomenFamily"]
# selector = SelectKBest(f_classif, k=5)
# selector.fit(csv_reader[predictors], csv_reader["Survived"])
# scores = -np.log10(selector.pvalues_)
# plt.bar(range(len(predictors)), scores)
# plt.xticks(range(len(predictors)), predictors, rotation='vertical')
# plt.show()

# Collect y
y = np.array(csv_reader["Survived"])
print "y collected successfully"

# Collect X
headers = list(csv_reader.columns.values)
headers.remove("Name")
headers = ["Pclass", "Sex", "Fare", "Title", "WomenFamily"]
num_features = len(headers)
num_examples = len(getattr(csv_reader, headers[0]))
for i in range(num_examples):
    X.append(map(float, csv_reader[headers].iloc[i, :]))
X = np.array(X)
print "X collected successfully"

# Shuffle data
X, X_cv, y, y_cv = train_test_split(X, y, test_size=0.0, random_state=42)
print "Data shuffled successfully"

# # Choosing regularized parameter
# parameters = [0.1, 0.3, 0.5, 0.6, 0.7, 0.9, 1]
# for p in parameters:
#     net = neural_network.NeuralNetwork(X, y, 2, num_features, len(X), 25, p)
#     net.train(100)
#     acc = 0
#     for i in range(len(X_cv)):
#         pred = net.interpret_result(net.predict(X_cv[i]))
#         if pred == y[i]:
#             acc += 1
#     print "Accuracy for parameter %f is %f%%" % (p, float(acc)/len(X)*100)


# Processing the test set
csv_test = pandas.read_csv("test.csv", usecols=[0, 1, 2, 3, 4, 5, 6, 8, 10])
headers_test = list(csv_test.columns.values)
m = len(getattr(csv_test, headers_test[0]))
X_test = []
csv_test["Family"] = csv_test["SibSp"] + csv_test["Parch"]
csv_test["NameLen"] = csv_test["Name"].apply(lambda x: len(x))
csv_test["Title"] = csv_test["Name"].apply(lambda x: title_mapping[get_title(x)])
for i in range(len(csv_test["Name"])):
    collect_family_id(csv_test["Name"][i], csv_test["Family"][i])
csv_test["FamilyID"] = map(lambda x: x.split(",")[0], csv_test["Name"])
IDs = []
for i in range((len(csv_test["Name"]))):
    IDs.append(FAMILY_IDs[csv_test["FamilyID"][i]+str(csv_test["Family"][i])])
csv_test["FamilyID"] = IDs
count_women_in_family(csv_test)
csv_test["WomenFamily"] = csv_test["FamilyID"].apply(lambda x: WOMEN_IN_FAMILY[x])
csv_test["Age"] = csv_test["Age"].fillna(csv_reader["Age"].median())
csv_test["Fare"] = csv_test["Fare"].fillna(csv_test["Fare"].median())
csv_test.loc[csv_test["Sex"] == "male", "Sex"] = 1
csv_test.loc[csv_test["Sex"] == "female", "Sex"] = 2
csv_test["Embarked"] = csv_test["Embarked"].fillna("S")
csv_test.loc[csv_test["Embarked"] == "S", "Embarked"]= 1
csv_test.loc[csv_test["Embarked"] == "C", "Embarked"] = 2
csv_test.loc[csv_test["Embarked"] == "Q", "Embarked"] = 3
headers_test = list(csv_test.columns.values)
headers_test.remove("Name")
headers_test = ["Pclass", "Sex", "Fare", "Title", "WomenFamily"]
print "Test set processed successfully"

# Collect X_test
for i in range(m):
    X_test.append(map(float, csv_test[headers_test].iloc[i, :]))
X_test = np.array(X_test)
print "X test collected successfully"

# Prepocess X and X_test
scaler = preprocessing.StandardScaler().fit(X)
X = scaler.transform(X)
X_test = scaler.transform(X_test)
normalizer = preprocessing.Normalizer().fit(X)
X = normalizer.transform(X)
X_test = normalizer.transform(X_test)
print "Data set and test set scaled and normalized successfully"

# y_test = svm(X, y, X_test)

# Train nn
net = neural_network.NeuralNetwork(X, y, 2, num_features, num_examples, 25, 0.1)
time0 = time.time()
net.train(200)
print "Data trained successfully in %f" % (time.time()-time0)

# Evaluate
accuracy = 0
for i in range(num_examples):
    out = net.predict(X[i])
    pred = net.interpret_result(out)
    if y[i] == pred:
        accuracy += 1
print "Accuracy is %f%%" % (float(accuracy)/num_examples*100)
print "Evaluation done"

# Perform on test set
y_test = []
for i in range(m):
    out = net.interpret_result(net.predict(X_test[i]))
    if out == 2:
        y_test.append(0)
    else:
        y_test.append(out)
t0 = time.time()
print "Prediction done successfully in %f" % (time.time()-t0)

# Write to CSV
final_result = pandas.DataFrame({
        "PassengerId": csv_test["PassengerId"],
        "Survived": y_test
    })
final_result.to_csv("kaggle.csv", index=False)
print "Done everything in %f" % (time.time()-begin)