


import numpy as np
import pandas as pd

data=pd.read_csv("./../../test.csv")

import numpy as np

tani = np.load("tanimoto_matrix.npy")

print(tani.shape)

tani[tani < 0.8 ] = 0
tani[tani >= 0.8] = 1

np.save("adjacency_matrix.npy",tani)


print(data.shape)

print("cutoff :",0.8)

r = np.load("adjacency_matrix.npy")

rows=r.shape[0]


a1=np.zeros(rows)
a2=np.zeros(rows)
a3=np.zeros(rows)
a4=np.zeros(rows)
a5=np.zeros(rows)
a6=np.zeros(rows)
a7=np.zeros(rows)
a8=np.zeros(rows)
a9=np.zeros(rows)
a10=np.zeros(rows)

for i in range(rows):
  c1=0; c2=0; c3=0; c4=0; c5=0; c6=0; c7=0; c8=0; c9=0; c10=0
  
  for j in range(rows):
    # sm-sm
    if r[i,j]==1 and data['stability-class'][j]==1 and data["magnetic-class"][j]==1 and data['stability-class'][i]==1 and data["magnetic-class"][i]==1 : 
      c1+=1
    # sm-sn
    if r[i,j]==1 and data['stability-class'][j]==1 and data["magnetic-class"][j]==0 and data['stability-class'][i]==1 and data["magnetic-class"][i]==1 : 
      c2+=1
    # sm-um
    if r[i,j]==1 and data['stability-class'][j]==0 and data["magnetic-class"][j]==1 and data['stability-class'][i]==1 and data["magnetic-class"][i]==1 : 
      c3+=1
    # sm-un
    if r[i,j]==1 and data['stability-class'][j]==0 and data["magnetic-class"][j]==0 and data['stability-class'][i]==1 and data["magnetic-class"][i]==1 : 
      c4+=1
    # sn-sn
    if r[i,j]==1 and data['stability-class'][j]==1 and data["magnetic-class"][j]==0 and data['stability-class'][i]==1 and data["magnetic-class"][i]==0 : 
      c5+=1
    # sn-um
    if r[i,j]==1 and data['stability-class'][j]==0 and data["magnetic-class"][j]==1 and data['stability-class'][i]==1 and data["magnetic-class"][i]==0 : 
      c6+=1
    # sn-un
    if r[i,j]==1 and data['stability-class'][j]==0 and data["magnetic-class"][j]==0 and data['stability-class'][i]==1 and data["magnetic-class"][i]==0 : 
      c7+=1
    # um-um
    if r[i,j]==1 and data['stability-class'][j]==0 and data["magnetic-class"][j]==1 and data['stability-class'][i]==0 and data["magnetic-class"][i]==1 : 
      c8+=1
    # um-un
    if r[i,j]==1 and data['stability-class'][j]==0 and data["magnetic-class"][j]==0 and data['stability-class'][i]==0 and data["magnetic-class"][i]==1 : 
      c9+=1
    # un-un
    if r[i,j]==1 and data['stability-class'][j]==0 and data["magnetic-class"][j]==0 and data['stability-class'][i]==0 and data["magnetic-class"][i]==0 : 
      c10+=1

  a1[i]=c1
  a2[i]=c2
  a3[i]=c3
  a4[i]=c4
  a5[i]=c5
  a6[i]=c6
  a7[i]=c7
  a8[i]=c8
  a9[i]=c9
  a10[i]=c10



sm_sm= np.sum(a1)
sm_sn = np.sum(a2)
sm_um = np.sum(a3)
sm_un = np.sum(a4)
sn_sn = np.sum(a5)
sn_um = np.sum(a6)
sn_un = np.sum(a7)
um_um = np.sum(a8)
um_un = np.sum(a9)
un_un = np.sum(a10)



print("sm_sm, sm_sn, sm_um, sm_un, sn_sn, sn_um, sn_un, um_um, um_un, un_un :",sm_sm, sm_sn, sm_um, sm_un, sn_sn, sn_um, sn_un, um_um, um_un, un_un)













