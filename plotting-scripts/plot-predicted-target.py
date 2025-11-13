



import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

sb.set_style("darkgrid")
plt.rc("axes", titlesize=8)
plt.rc("axes", labelsize=6)
plt.rc("xtick", labelsize=4)
plt.rc("ytick", labelsize=4)
plt.rc("legend", fontsize=7)
plt.rc("font", size=8)

# dataset
y_test = np.load("magnetization_test.npy")
y_pred_test = np.load("magnetization_pred_test.npy")

fig = plt.figure(figsize=(3,2), dpi=200)

plt.scatter(y_test,y_pred_test,marker=".",s=0.5,color="orange",label="$R^2$=0.87")
plt.plot(y_test, y_test,color="black",linestyle="--",linewidth=0.4)
plt.xlabel("Actual magnetization ($\mu_B/\AA^3$)")
plt.ylabel("Predicted magnetization ($\mu_B/\AA^3$)")

#plt.xlabel("Actual hform (meV/atom)")
#plt.ylabel("Predicted hform (meV/atom)")


plt.legend()
plt.tight_layout()
plt.savefig("magnetization-regression.jpg",bbox_inches="tight",dpi=1200)
plt.show()








