

# Here we will add material tag : elemental, binary, ternary


import pandas as pd
import ase.io


data = pd.read_csv("result.csv")
ID = data["ID"].tolist()
validity = data["validity"].tolist()
material = data["material"].tolist()
Nmat = len(ID)

mat_type = []
TM = []

for i in range(Nmat):
    mat_id = ID[i]
    val = validity[i]
    if val == "valid" :
        compound = "./cif-files/valid/" + mat_id + ".cif"
    if val == "moderate" :
        compound = "./cif-files/moderate/" + mat_id + ".cif"
    if val == "invalid" :
        compound = "./cif-files/invalid/" + mat_id + ".cif"
    
    atoms = ase.io.read(compound)
    z1 = atoms.get_atomic_numbers()
    z = set(z1)
    count = len(z)
    if count == 1 :
        mat_type.append("elemental") 
    if count == 2 :
        mat_type.append("binary")
    if count == 3 :
        mat_type.append("ternary")
        
    TM_list = [24,25,26,27,28]
    m = 0
    for val in z1 :
        if val in TM_list :
           m += 1
    if m != 0 :
        TM.append("yes")
    else :
        TM.append("no")
       


data["type"] = mat_type
data["3d-TM"] = TM
data.to_csv("result-type-3d-TM.csv",index=False)




