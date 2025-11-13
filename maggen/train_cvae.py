import numpy as np
import pandas as pd
from cvae import VAE
from utils import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



# import the dataset
data_train = pd.read_csv("./../train.csv")
data_test = pd.read_csv("./../test.csv")

x_train = np.load("./../point-cloud-train.npy")
x_test = np.load("./../point-cloud-test.npy")

hform_train = data_train["hform"].tolist()
hform_test = data_test["hform"].tolist()

magnetization_train = data_train["magnetization"].tolist()
magnetization_test = data_test["magnetization"].tolist()


hform_train = np.asarray(hform_train).reshape(-1,1)
hform_test = np.asarray(hform_test).reshape(-1,1)

magnetization_train = np.asarray(magnetization_train).reshape(-1,1)
magnetization_test = np.asarray(magnetization_test).reshape(-1,1)

x_train, scaler_x_train = minmax(x_train)
x_test, scaler_x_test = minmax(x_test)

scaler_hform = MinMaxScaler()
hform_train = scaler_hform.fit_transform(hform_train)
hform_test = scaler_hform.fit_transform(hform_test)

scaler_magnetization = MinMaxScaler()
magnetization_train = scaler_magnetization.fit_transform(magnetization_train)
magnetization_test = scaler_magnetization.fit_transform(magnetization_test)


learning_rate = 5e-4
batch_size = 256
num_epochs = 300

vae = VAE(input_shape=(144,3),latent_dim=256)
 
vae.summary()   
vae.compile(learning_rate)
vae.train(x_train,x_test,hform_train,hform_test,magnetization_train,magnetization_test,batch_size,num_epochs)

# saving the VAE model
vae.save("vae-model")
vae2 = VAE.load("vae-model")
vae2.summary()

# load the VAE model
vae = VAE.load("vae-model")
latent_reps_train, hform_pred_train, magnetization_pred_train, recons_x_train = vae.reconstruct(x_train)
latent_reps_test, hform_pred_test, magnetization_pred_test, recons_x_test = vae.reconstruct(x_test)

recons_x_train = inv_minmax(recons_x_train,scaler_x_train)
recons_x_train[recons_x_train < 0.05 ] = 0

recons_x_test = inv_minmax(recons_x_test,scaler_x_test)
recons_x_test[recons_x_test < 0.05 ] = 0

# saving the files
np.save("latent_reps_train.npy",latent_reps_train)
np.save("latent_reps_test.npy",latent_reps_test)
np.save("recons_x_train.npy",recons_x_train)
np.save("recons_x_test.npy",recons_x_test)

hform_train = scaler_hform.inverse_transform(hform_train)
hform_test = scaler_hform.inverse_transform(hform_test)
hform_pred_train = scaler_hform.inverse_transform(hform_pred_train)
hform_pred_test = scaler_hform.inverse_transform(hform_pred_test)

magnetization_train = scaler_magnetization.inverse_transform(magnetization_train)
magnetization_test = scaler_magnetization.inverse_transform(magnetization_test)
magnetization_pred_train = scaler_magnetization.inverse_transform(magnetization_pred_train)
magnetization_pred_test = scaler_magnetization.inverse_transform(magnetization_pred_test)

np.save("hform_test.npy",hform_test)
np.save("hform_pred_test.npy",hform_pred_test)
np.save("magnetization_test.npy",magnetization_test)
np.save("magnetization_pred_test.npy",magnetization_pred_test)


def get_mae_r2(y_true,y_predicted):
    mae = mean_absolute_error(y_true,y_predicted)
    r2 = r2_score(y_true,y_predicted)
    return mae,r2

hform_mae, hform_r2 = get_mae_r2(hform_test,hform_pred_test)
magnetization_mae, magnetization_r2 = get_mae_r2(magnetization_test,magnetization_pred_test)

print("Test set results")
print("hform prediction")
print("MAE :",hform_mae)
print("R2 :",hform_r2)
print("magnetization prediction")
print("MAE :",magnetization_mae)
print("R2 :",magnetization_r2)









