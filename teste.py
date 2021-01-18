import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import SGDRegressor
#https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html
import numpy as np

lb = LabelEncoder() 

casasDataset = pd.read_csv("casas_aluguel.csv")

casasDataset.animal = lb.fit_transform(casasDataset.animal)
casasDataset.furniture = lb.fit_transform(casasDataset.furniture)
casasDataset.city = lb.fit_transform(casasDataset.city)
casasDataset.floor = lb.fit_transform(casasDataset.floor)

x = casasDataset.iloc[:, :-1]
y = casasDataset.iloc[:, -1]
a,b = np.polyfit(x,y,1)



#floor                  10692 non-null object
#animal                 10692 non-null object
#furniture              10692 non-null object
