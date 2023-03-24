import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

dataset = pd.read_csv('Data Covid19.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x[:,1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

print(x)
print(y)

labelencoder_X = LabelEncoder()
x[:, 0] = labelencoder_X.fit_transform(x[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
x = onehotencoder.fit_transform(x).toarray()
print('X = ', x)

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
print('y = ', y)