

# Here we will plot the latent space


import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


sb.set_style("darkgrid")
plt.rc("axes", titlesize=8)
plt.rc("axes", labelsize=6)
plt.rc("xtick", labelsize=4)
plt.rc("ytick", labelsize=4)
plt.rc("legend", fontsize=7)
plt.rc("font", size=8)
fig = plt.figure()

fig = plt.figure(figsize=(3,2), dpi=200)

# importing the datasets

data_test = pd.read_csv("./../test.csv")
stability_class = data_test["stability-class"].tolist()
hform_test = data_test["hform"].tolist()
magnetization_test = data_test["magnetization"].tolist()
magnetic_class = data_test["magnetic-class"].tolist()


all_class = []
for i in range(len(hform_test)):
    sta = stability_class[i]
    mag = magnetic_class[i]
    if sta == 0 and mag == 0 :
        all_class.append(1)
    if sta == 0 and mag == 1 :
        all_class.append(2)
    if sta == 1 and mag == 0 :
        all_class.append(3)
    if sta == 1 and mag == 1 :
        all_class.append(4)



# load latent space
latent_space_test = np.load("latent_reps_test.npy")


# dimensionality reduction of latent space by PCA
pca = PCA(n_components=2)
pca.fit(latent_space_test)
variance_list = pca.explained_variance_ratio_
variance = np.sum(variance_list)
print("variance explained :",variance)
reduced_latent_space = pca.transform(latent_space_test)


fig = plt.figure(figsize=(3,2), dpi=200)
plt.scatter(reduced_latent_space[:,0],
            reduced_latent_space[:,1],
            cmap="viridis",
            c=all_class,
            alpha=0.8,
            s=0.1)
plt.colorbar()
plt.xlabel("$Z_1$")
plt.ylabel("$Z_2$")
plt.savefig("z-all-class-new.jpg",bbox_inches="tight",dpi=1200)





