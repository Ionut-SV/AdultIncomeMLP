import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn import neural_network

data = pd.read_csv('adultdata.csv')
data_test = pd.read_csv('adultest.csv')

data = data.dropna()
le = LabelEncoder()

for col in data.columns:
    if data[col].dtype == 'object':
        le.fit(data[col])
        data[col] = le.transform(data[col])

data_test = data_test.dropna()

for col in data_test.columns:
    if data_test[col].dtype == 'object':
        le.fit(data_test[col])
        data_test[col] = le.transform(data_test[col])

data['salary'] = data['salary'].astype('category')
data_test['salary'] = data_test['salary'].astype('category')

category_train = np.array(data['salary'].cat.codes.values).reshape(-1, 1)
category_test = np.array(data_test['salary'].cat.codes.values).reshape(-1, 1)

date_train = np.array(data.drop(['salary'], axis=1))
date_test = np.array(data_test.drop(['salary'], axis=1))

etichete_train = np.array(data['salary'].values)
etichete_test = np.array(data_test['salary'].values)

size_train = len(data['salary'])
size_test = len(data_test['salary'])

train_size = int(0.66 * size_train)
test_size = int(0.33 * size_test)

# IMPARTIRE IN TRAIN SI TEST
category_train = category_train[:train_size]
category_test = category_test[test_size:]

date_train = date_train[:train_size]
date_test = date_test[test_size:]

etichete_train = etichete_train[:train_size]
etichete_test = etichete_test[test_size:]

# CREARE SI ANTRENARE MLP
lr = [0.1, 0.01]
nr_col = 14

for i in range(2):
    for j in range(3):
        nr_strat = i + 1
        if nr_strat == 1:
            layer1 = nr_col // (j + 1)
            clf = neural_network.MLPClassifier(hidden_layer_sizes=(layer1),
                                               learning_rate_init=lr[i])
        elif nr_strat == 2:
            layer1 = nr_col // (j + 1)
            layer2 = layer1 // (j + 1)
            clf = neural_network.MLPClassifier(hidden_layer_sizes=(layer1, layer2),
                                               learning_rate_init=lr[i])
        clf.fit(np.append(category_train, date_train, axis=1), etichete_train)
        predictii = clf.predict(np.append(category_test, date_test, axis=1))
        acc = accuracy_score(etichete_test, predictii)
        print("Acuratetea este =",acc*100,'%')