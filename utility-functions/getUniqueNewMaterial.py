

import pandas as pd

data1 = pd.read_csv("result-type-3d-TM.csv")
data2 = pd.read_csv("material-mp-novo-nova.csv")

print("Total generated materials :",data1.shape[0])


data1 = data1.drop_duplicates(subset=["material","validity"], keep="first")
print("After removing duplicates, total generated materials :",data1.shape[0])



data1 = data1[(data1["validity"]=="moderate") & (data1["hform"]<=-500) & (data1["magnetization"]>=0.5) & (data1["type"]=="ternary")]


print(data1.shape)

parent_material = data2["material"].tolist()
new_material = data1["material"].tolist()
new_material_id = data1["ID"].tolist()


new_id = []

m = 0
for i in range(len(new_material)) :
    mat = new_material[i]
    if mat not in parent_material :
        new_id.append(new_material_id[i])
        m += 1


print("# of unique materials :",m)

data_final = data1.loc[data1["ID"].isin(new_id)]
data_final.to_csv("ternary.csv",index=False)












