#%%
import numpy as np 
import matplotlib.pyplot as plt

y = np.load("original.npy")
y1 = np.load("shifted.npy")
plt.imshow(y[0][0:196])
plt.show()

plt.imshow(y1[0][0:196])
plt.show()

print(np.linalg.norm(y))
print(np.linalg.norm(y1))

# TODO check original swin behavior as well 
# check content of each token and compare 

# %%
