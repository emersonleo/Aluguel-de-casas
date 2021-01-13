import pandas as pd
from sklearn.preprocessing import LabelEncoder

lb = LabelEncoder() 

casasDataset = pd.read_csv("casas_aluguel.csv")

casasDataset.animal = lb.fit_transform(casasDataset.animal)
casasDataset.furniture = lb.fit_transform(casasDataset.furniture)
casasDataset.city = lb.fit_transform(casasDataset.city)
casasDataset.floor = lb.fit_transform(casasDataset.floor)

print(casasDataset.groupby(by='city').size())

#floor                  10692 non-null object
#animal                 10692 non-null object
#furniture              10692 non-null object
