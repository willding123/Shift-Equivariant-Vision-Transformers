#%%
import numpy as np 
import matplotlib.pyplot as plt

y = np.load("original.npy")
y1 = np.load("shifted.npy")
plt.imshow(y[0][0:196])
plt.imshow(y[1][0:196])
plt.show()

# %%
