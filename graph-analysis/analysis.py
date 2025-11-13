


import numpy as np
import pandas as pd

    
def get_analysis(matrix,cutoff,data):
    tani = matrix
    cutoff = cutoff
    tani[tani < cutoff ] = 0
    tani[tani >= cutoff] = 1
    data = data
    stab = data["stability-class"].tolist()
    Nmat = data.shape[0]

    ss = []; su = []; us = []; uu = []

    for i in range(Nmat):
        for j in range(i+1,Nmat):
            if tani[i,j] == 1 and stab[i] == 1 and stab[j] == 1 :
                ss.append(1)
            if tani[i,j] == 1 and stab[i] == 1 and stab[j] == 0 :
                su.append(1)
            if tani[i,j] == 1 and stab[i] == 0 and stab[j] == 1 :
                us.append(1)
            if tani[i,j] == 1 and stab[i] == 0 and stab[j] == 0 :
                uu.append(1)

    ss_count = np.sum(ss)
    su_count = np.sum(su)
    us_count = np.sum(us)
    uu_count = np.sum(uu)
    off_diagonal = su_count + us_count

    result = np.zeros((2,2))
    result[0,0] = ss_count
    result[0,1] = result[1,0] = off_diagonal
    result[1,1] = uu_count
    return result


def get_fraction_of_wrong_connections(conf_mat):
    conf_mat = conf_mat
    total = conf_mat[0,0] + conf_mat[0,1] + conf_mat[1,1]
    wrong = conf_mat[0,1]
    fraction = np.around(wrong/total,3)
    return 100*fraction





cutoffs = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
#cutoffs = [0.45]

for i in range(len(cutoffs)):
    cutoff = cutoffs[i]
    matrix = np.load("tanimoto_matrix.npy")
    data = pd.read_csv("./../../test.csv")
    result = get_analysis(matrix,cutoff,data)
    fraction = get_fraction_of_wrong_connections(result)
    print("cutoff :",cutoff)
    print(result)
    print("Percentage of wrong connections :",fraction)
    print("======================") 














