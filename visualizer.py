#%%
import numpy as np 
import matplotlib.pyplot as plt
from tqdm import tqdm
#%%

np.random.seed(123)

y = np.load("original.npy")
y1 = np.load("shifted.npy")

plt.imshow(y[0][0:196])
plt.show()

plt.imshow(y1[0][0:196])
plt.show()

print(np.linalg.norm(y))
print(np.linalg.norm(y1))


# pick two random tokens from same image y[0]
idx = np.random.randint(0,3136,2)
avg = np.mean(y[0][idx[0]]); avg1 = np.mean(y[0][idx[1]])
token = y[0][idx[0]].reshape(12,16) ; token1 = y[0][idx[1]].reshape(12,16)
print(avg, avg1)
token = np.where(token > avg , 255,  0); token1 = np.where(token1 > avg1, 255, 0)
plt.imshow(token)
plt.show()
plt.imshow(token1)
plt.show()
#%% 
# pick the same indexed token from both y and y1 
idx = np.random.randint(0,3136)
avg = np.mean(y[0][idx]); avg1 = np.mean(y1[0][idx])
token = y[0][idx].reshape(12,16); token1 = y1[0][idx].reshape(12,16)
print(avg, avg1)
token = np.where(token > avg , 255,  0); token1 = np.where(token1 > avg1, 255, 0)
plt.imshow(token)
plt.show()
plt.imshow(token1)
plt.show()

# TODO check original swin behavior as well 
# check content of each token and compare 

        
# %%


# %%
def find_similar(anchor:np.array, y_search:np.array, threshold:float):
    ''' Find matching of anchor in y_search matrix
    '''
    matches = []
    for i in range(y_search.shape[1]):
        if np.linalg.norm(anchor - y_search[0][i]) < threshold:
            matches.append(i)
    return matches

for i in tqdm(range(3136)):
    anchor = y[0][i]
    if find_similar(anchor, y1, 0.1):
        print(i)

# Find binary thresholded match
y_mean = np.mean(y)
y1_mean = np.mean(y1)
y_binary =  np.where(y > y_mean, 255, 0)
y1_binary = np.where(y1 > y1_mean, 255, 0)

for i in tqdm(range(3136)):
    anchor = y_binary[0][i]
    if find_similar(anchor, y1_binary, 0.1):
        print(i)


# %%

# %%
