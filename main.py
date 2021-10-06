import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.datasets import load_iris

file = pd.read_csv("D:\\Progs\\ISIS\\Lab3\\Car_Data.dat")
data = []
labels = []
for i in range(file.size):
    line = file.values[i][0]
    line = line.replace(';',' ')
    object = line.split()
    #data.append(object[1:-1])
    data.append(object[1:])
    labels.append(object[-1])

data = np.array(data, dtype=float)
np.random.shuffle(data)
labels = data[:,-1]
data = data[:,:-1]
data_separator = 50
modelKN = KNeighborsClassifier(n_neighbors=3)
modelKN.fit(data[:data_separator],labels[:data_separator])
predictKN = modelKN.predict(data[data_separator:])
print("KNeighborsClassifier: ",accuracy_score(predictKN,labels[data_separator:]))
modelSVC = SVC()
modelSVC.fit(data[:data_separator],labels[:data_separator])
predictSVC = modelSVC.predict(data[data_separator:])
print('SVC: ',accuracy_score(predictSVC,labels[data_separator:]))

# ref_data = load_iris()
#
# ref_data.drop('Id', axis=1, inplace=True)
#
# X = ref_data.iloc[:,:-1].values
#
# y = ref_data['Species']
#
# X = ref_data.iloc['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm']
