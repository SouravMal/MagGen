

import pandas as pd


data = pd.read_csv("binary.csv")
Nmat = data.shape[0]


print("Total number of unique ternary materials :",Nmat)

data = data[(data["3d-TM"]=="yes") & (data["magnetization"]>=0.8)]

print("Number of materials containing 3d TM :",data.shape[0])

print(data.loc[1000:1200,:])
