

import numpy as np

tani = np.load("tanimoto_matrix.npy")

print(tani.shape)

tani[tani < 0.4 ] = 0
tani[tani >= 0.4] = 1

np.save("adjacency_matrix_threshold_4.npy",tani)


