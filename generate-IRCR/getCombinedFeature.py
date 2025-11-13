

import numpy as np
import pandas as pd
from getPCR_new import PCR


z_max = 94
n_sites = 20

data = pd.read_csv("train.csv")
ID = data["ID"].tolist()
Nmat = len(ID)

# to store pcr of each material 
feature = []  
for i in range(Nmat):
    mat_id = ID[i]
    compound = "./../cif-files-MP-Novo-Nova/"+mat_id+".cif"
    pcr = PCR(compound,z_max,n_sites)
    pcr_mat = pcr.get_pcr()
    feature.append(pcr_mat)
    
feature = np.asarray(feature)
np.save("point-cloud-train.npy",feature)




