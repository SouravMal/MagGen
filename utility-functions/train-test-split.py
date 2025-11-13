

import pandas as pd
from sklearn.model_selection import train_test_split as tts



data = pd.read_csv("MP-Novo-Nova.csv")
data = data[(data["magnetization"]>=5e-9/11.649)]

# remove rows if any "nan" is there
data = data.dropna().reset_index(drop=True)
print("after removing nan :",data.shape)


hform = data["hform"].tolist()
magnetization = data["magnetization"].tolist()

stability_class = []
for ene in hform :
    if ene <= 0 :
        stability_class.append(1)
    else :
        stability_class.append(0)

magnetic_class = []
for mom in magnetization :
    if mom >= 0.1/11.649 :
        magnetic_class.append(1)
    else :
        magnetic_class.append(0)

data["stability-class"] = stability_class
data["magnetic-class"] = magnetic_class


train, test = tts(data,test_size=0.2,random_state=1,shuffle=True,stratify=magnetic_class)

print("training set size :",train.shape)
print("test set size :",test.shape)

train.to_csv("train.csv",index=False)
test.to_csv("test.csv",index=False)

print("done...")



