import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv('titan.csv', index_col='PassId')

X = np.empty([0, 4])
y = np.empty(0)
idx = 0
for index, row in data.iterrows():
	if pd.isna(row['Pclss']) or pd.isna(row['Far']) or pd.isna(row['Ag']) or pd.isna(row['Sx']):
		continue

	X = np.resize(X, (idx + 1, 4))
	y = np.resize(y, idx + 1)

	X[idx][0] = row['Pclss']
	X[idx][1] = row['Far']
	X[idx][2] = row['Ag']
	if row['Sx'] == 'male':
		X[idx][3] = 1
	else:
		X[idx][3] = 0
	y[idx] = row['Srvivd']

	idx += 1

clf = DecisionTreeClassifier(random_state=242)
clf.fit(X, y)
importance = clf.feature_importances_
