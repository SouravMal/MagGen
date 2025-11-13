

import pandas as pd
import numpy as np
from invertPCR import InvertPCR


# import the dataset
x_train = np.load("./../point-cloud-train.npy")
x_test = np.load("./../point-cloud-test.npy")
recons_x_train = np.load("recons_x_train.npy")
recons_x_test = np.load("recons_x_test.npy")

data_test = pd.read_csv("./../test.csv")

def get_MAE_fractional_coordinates(x,recons_x) :
    z_max = 94
    n_sites = 20
    Nmat = x.shape[0]
   
    MAE = []
    MSE = []
    for i in range(Nmat):
        pcr = x[i,:]
        pcr_recons = recons_x[i,:]
        ipcr = InvertPCR(pcr,z_max,n_sites)
        ipcr_recons = InvertPCR(pcr_recons,z_max,n_sites)

        basis_mat = ipcr.get_basis_matrix()
        basis_mat_recons = ipcr_recons.get_basis_matrix()
        basis_mat_recons[basis_mat_recons <= 0.05] = 0        

        abs_err = abs(basis_mat - basis_mat_recons)
        mean_abs_err = np.sum(abs_err)/60 #(n_sites*3)
        mean_squared_err = np.sum(abs_err**2)/60#(n_sites*3)
        MAE.append(mean_abs_err)
        MSE.append(mean_squared_err)

    mae_val = np.sum(MAE)/Nmat
    mse_val = np.sum(MSE)/Nmat
    rmse_val = mse_val**0.5 
    return [mae_val,rmse_val]

mae_train = get_MAE_fractional_coordinates(x_train,recons_x_train)
mae_test = get_MAE_fractional_coordinates(x_test,recons_x_test)

print("")
print("MAE, RMSE of fractional coordinates :")
print("training set :",mae_train)
print("test set :",mae_test)


def get_atomic_numbers(img):
    z_max = 94
    n_sites = 20
    ipcr = InvertPCR(img,z_max,n_sites)
    z_list = ipcr.get_unique_atomic_numbers()
    return z_list



def get_accuracy_z_reconstruction(x,x_recons):
    Nmat = x.shape[0]
    z_recons = []
    for i in range(Nmat):
       pcr = x[i,:]
       pcr_recons = x_recons[i,:]
       z1 = get_atomic_numbers(pcr)
       z2 = get_atomic_numbers(pcr_recons)
       if z1 == z2 :
          z_recons.append(1) # 1 : successful
       if z1 != z2 :
          z_recons.append(0) # 0 : failed
    accuracy = z_recons.count(1)/Nmat
    return accuracy



train_accuracy = get_accuracy_z_reconstruction(x_train,recons_x_train)
test_accuracy = get_accuracy_z_reconstruction(x_test,recons_x_test)

print("")
print("accuracy of Z reconstruction :")
print("training :",train_accuracy)
print("test :",test_accuracy)
print("")



# defining a function to calculate MAPE

def MAPE(y_true,y_pred) :
    N = len(y_true)
    abs_relative_error = 0
    for i in range(N) :
        y = y_true[i]
        y_hat = y_pred[i]
        abs_relative_error += abs((y-y_hat)/y)
    mape = abs_relative_error*100/N
    return mape


def RMSE(y_true,y_pred) :
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    N = len(y_true)
    mse = np.sum((y_true - y_pred)**2)/N
    rmse = np.round(mse**0.5,2)
    return rmse

def MAE(y_true,y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    N = len(y_true)
    mae = np.sum(abs(y_true - y_pred))/N
    return mae

def get_mape_rmse_mae_abc_abg(x,recons_x):
    z_max = 94
    n_sites = 20
    Nmat = x.shape[0]
    a=[]; b=[]; c=[]
    al=[]; be=[]; ga=[]
    a_hat=[]; b_hat=[]; c_hat=[]
    al_hat=[]; be_hat=[]; ga_hat=[]

    for i in range(Nmat):
        pcr = x[i,:]
        pcr_recons = recons_x[i,:]
        ipcr = InvertPCR(pcr,z_max,n_sites)
        ipcr_recons = InvertPCR(pcr_recons,z_max,n_sites)
        lattice_par = ipcr.get_lattice_parameters()
        lattice_par_recons = ipcr_recons.get_lattice_parameters()

        a.append(lattice_par[0])
        a_hat.append(lattice_par_recons[0])

        b.append(lattice_par[1])
        b_hat.append(lattice_par_recons[1])

        c.append(lattice_par[2])
        c_hat.append(lattice_par_recons[2])

        al.append(lattice_par[3])
        al_hat.append(lattice_par_recons[3])

        be.append(lattice_par[4])
        be_hat.append(lattice_par_recons[4])

        ga.append(lattice_par[5])
        ga_hat.append(lattice_par_recons[5])


    mape_a = MAPE(a,a_hat)
    mape_b = MAPE(b,b_hat)
    mape_c = MAPE(c,c_hat)
    mape_al = MAPE(al,al_hat)
    mape_be = MAPE(be,be_hat)
    mape_ga = MAPE(ga,ga_hat)


    rmse_a = RMSE(a,a_hat)
    rmse_b = RMSE(b,b_hat)
    rmse_c = RMSE(c,c_hat)
    rmse_al = RMSE(al,al_hat)
    rmse_be = RMSE(be,be_hat)
    rmse_ga = RMSE(ga,ga_hat)

    mae_a = MAE(a,a_hat)
    mae_b = MAE(b,b_hat)
    mae_c = MAE(c,c_hat)
    mae_al = MAE(al,al_hat)
    mae_be = MAE(be,be_hat)
    mae_ga = MAE(ga,ga_hat)

    mape_list = [mape_a,mape_b,mape_c,mape_al,mape_be,mape_ga]
    rmse_list = [rmse_a,rmse_b,rmse_c,rmse_al,rmse_be,rmse_ga]
    mae_list = [mae_a,mae_b,mae_c,mae_al,mae_be,mae_ga]
    return mape_list,rmse_list, mae_list


mape_train, rmse_train, mae_train  = get_mape_rmse_mae_abc_abg(x_train,recons_x_train)
mape_test, rmse_test, mae_test  = get_mape_rmse_mae_abc_abg(x_test,recons_x_test)


print("")
print("MAPE of a,b,c,alpha,beta,gamma :")
print("training set :",mape_train)
print("test set :",mape_test)
print("")
print("RMSE of a,b,c,alpha,beta,gamma :")
print("training set :",rmse_train)
print("test set :",rmse_test)
print("")
print("MAE of a,b,c,alpha,beta,gamma :")
print("training set :",mae_train)
print("test set :",mae_test)
print("")




