

# Here we will create the Tanimoto matrix of N number of materials from their latent representations.
# The matrix size will be NxN.
# T_ij is the Tanimoto coefficient between i and j-th materials.


import numpy as np


latent_reps = np.load("./../latent_reps_test.npy")


# calculating tanimoto coefficient 
def tanimoto_coeff(x1,x2):
    coeff = np.dot(x1,x2)/(np.dot(x1,x1) + np.dot(x2,x2) - np.dot(x1,x2))
    return coeff


Nmat = latent_reps.shape[0]
T = np.zeros((Nmat,Nmat))

for i in range(Nmat):
    for j in range(Nmat):
        x1 = latent_reps[i,:]
        x2 = latent_reps[j,:]
        T[i,j] = tanimoto_coeff(x1,x2)


np.save("tanimoto_matrix.npy",T)
