
# Script for new material generation
# Author : Sourav Mal
# Date : 05/07/2023
#=============================================================


import ase.io
import shutil
import pandas as pd
import numpy as np
from cvae import VAE
from utils import *
from invertPCR import InvertPCR
from sklearn.preprocessing import MinMaxScaler
from analysis_structure_validity import get_validity


# import the dataset
data_train = pd.read_csv("train.csv")
ID = data_train["ID"].tolist()
x_train = np.load("point-cloud-train.npy")
material = data_train["material"].tolist()

hform = data_train["hform"].tolist()
magnetization = data_train["magnetization"].tolist()

hform_train = data_train["hform"].tolist()
magnetization_train = data_train["magnetization"].tolist()


def get_index_parent_material(target_Ef,target_Ms,hform,magnetization):
    hform = hform
    magnetization = magnetization
    target_Ef = target_Ef
    target_Ms = target_Ms
    
    index_constraint = []
    Nmat = len(hform)
    for i in range(Nmat):
        if hform[i] <= target_Ef and magnetization[i] >= target_Ms/11.649 :
            index_constraint.append(i)
    Nparent = len(index_constraint)
    index_constraint = np.asarray(index_constraint).reshape(-1,1)
    index_constraint = np.squeeze(index_constraint)
    return Nparent,index_constraint


# scaling the data
hform_train = np.asarray(hform_train).reshape(-1,1)
magnetization_train = np.asarray(magnetization_train).reshape(-1,1)
x_train, scaler_x_train = minmax(x_train)
scaler_hform = MinMaxScaler()
hform_train_scaled = scaler_hform.fit_transform(hform_train)
scaler_magnetization = MinMaxScaler()
magnetization_train_scaled = scaler_magnetization.fit_transform(magnetization_train)

# design targets : Ef <= -500 meV/atom, Ms >= 0.5 T
target_Ef = -500
target_Ms = 0.5
Nparent, index_constraint = get_index_parent_material(target_Ef,target_Ms,hform,magnetization)

# latent space
latent_space = np.load("latent_reps_train.npy")
latent_space_parent = latent_space[index_constraint,:]


def generate_material(latent_space_parent,Lp):

    """
    Given the latent vectors of parent materials and having set the local
    perturbation scale, this function can generate new material with 
    its predicted hform (meV/atom) and magnetization (Tesla). 

    """

    z = latent_space_parent               # latent vector of parent material
    Lp = Lp                               # local perturbation scale
    noise = np.random.normal(0,1,z.shape) # random gaussian noise 
    z_noise = z + Lp*noise                # noisy latent vector
    # new material generation
    new_img = vae.getReconstructedImage(z_noise) 
    new_img = inv_minmax(new_img,scaler_x_train)
    new_img = np.round(new_img,4)
    new_img[new_img < 0.05] = 0  # replace all the numbers < 0.05 by 0
    # prediction of targets
    pred_hform = vae.getPredictedHform(z_noise)
    pred_hform = scaler_hform.inverse_transform(pred_hform) # unit : meV/atom
    pred_magmom = vae.getPredictedMagnetization(z_noise)
    pred_magmom = 11.649*scaler_magnetization.inverse_transform(pred_magmom) # unit : Tesla
    return new_img, pred_hform, pred_magmom
    

# load the vae model
vae = VAE.load("vae-model")

Lp = 1.0
Nperturb = 20  # number of perturbing instances
z_max = 94
n_sites = 20


print("")
print("Total number of parent materials :",Nparent)

m = 0

# empty lists to store information of generated materials
parent_material_id = []
parent_material_formula = []
parent_material_hform = []
parent_material_magmom = []

generated_material_id = []
generated_material_formula = []
generated_material_hform = []
generated_material_magmom = []
generated_material_comment = [] # comment regarding validity

for i in range(Nperturb):
    
    new_img, pred_hform, pred_magmom = generate_material(latent_space_parent,Lp)

    m += 1
    for j in range(Nparent):

        idx = index_constraint[j]
        parent_id = ID[idx]
        pcr = new_img[j,:]
        ipcr = InvertPCR(pcr,z_max,n_sites)
        pred_ene = pred_hform[j][0]
        pred_mom = pred_magmom[j][0]
        try :
            
            atoms = ipcr.get_atoms_object() # atoms object
            formula = ipcr.get_formula()    # material 
            z = atoms.get_atomic_numbers()
            distances = ipcr.get_distances() 
            comment = get_validity(z,distances)
            mat_id = parent_id + "-" +  str(m) + "-Lp-1" 
            cif_file = ase.io.write(mat_id+".cif",atoms)
            # appending in the lists
            parent_material_id.append(parent_id)
            parent_material_formula.append(material[idx])
            parent_material_hform.append(hform[idx])
            parent_material_magmom.append(11.649*magnetization[idx])

            generated_material_id.append(mat_id)
            generated_material_formula.append(formula)
            generated_material_hform.append(pred_ene)
            generated_material_magmom.append(pred_mom)
            generated_material_comment.append(comment)

            if comment == "valid" :
                src = "./" + mat_id + ".cif"
                dst = "./cif-files/valid"
                shutil.move(src,dst)
            if comment == "moderate" :
                src = "./" + mat_id + ".cif"
                dst = "./cif-files/moderate"
                shutil.move(src,dst)
            if comment == "invalid" :
                src = "./" + mat_id + ".cif"
                dst = "./cif-files/invalid"
                shutil.move(src,dst)
            
            print(formula,mat_id,comment)

        except Exception as e :
            print("Error :",e)            



# csv file
data_dict = {"ID":generated_material_id, "material":generated_material_formula,\
             "hform":generated_material_hform, "magnetization":generated_material_magmom,\
             "validity":generated_material_comment,\
             "parent-ID":parent_material_id, "parent-material":parent_material_formula,\
             "parent-hform":parent_material_hform, "parent-magnetization":parent_material_magmom}

df = pd.DataFrame(data_dict)
df.to_csv("result.csv",index=False)









