

import numpy as np
from invertPCR import InvertPCR

# import the dataset
x_train = np.load("./../point-cloud-train.npy")
x_test = np.load("./../point-cloud-test.npy")
recons_x_train = np.load("recons_x_train.npy")
recons_x_test = np.load("recons_x_test.npy")


def get_z_accuracy(x,recons_x) :
    z_max = 94
    n_sites = 20
    Nmat = x.shape[0]

    MAE = []
    MSE = []
    correct_z_recons = []
    correct_all_z_recons = []

    for i in range(Nmat):
        pcr = x[i,:]
        pcr_recons = recons_x[i,:]
        ipcr = InvertPCR(pcr,z_max,n_sites)
        ipcr_recons = InvertPCR(pcr_recons,z_max,n_sites)

        property_mat = ipcr.get_property_matrix()
        property_mat_recons = ipcr_recons.get_property_matrix()
        
        property_mat = np.rint(property_mat)
        property_mat_recons = np.rint(property_mat_recons)

        z = property_mat[0,:]
        z_recons = property_mat_recons[0,:]
        
        #site matrix
        site_mat = ipcr.get_site_matrix()
        site_mat_recons = ipcr_recons.get_site_matrix()
        site_mat = np.rint(site_mat)
        site_mat_recons = np.rint(site_mat_recons)
        
        val = np.array_equal(site_mat,site_mat_recons)
        if val == True :
           correct_all_z_recons.append(1)
 
        if list(z) == list(z_recons) :
            correct_z_recons.append(1)

    accuracy = len(correct_z_recons)/Nmat
    all_accuracy = len(correct_all_z_recons)/Nmat
    return [accuracy,all_accuracy]
            

train_accuracy = get_z_accuracy(x_train,recons_x_train)
test_accuracy = get_z_accuracy(x_test,recons_x_test)



print("")
print("z accuracy")
print("first number for three z , the second number for all z ")
print("")
print("training :",train_accuracy)
print("test :",test_accuracy)


