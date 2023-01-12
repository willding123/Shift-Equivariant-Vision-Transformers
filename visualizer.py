#%%
import numpy as np 
import matplotlib.pyplot as plt
from tqdm import tqdm
#%%

np.random.seed(123)

y = np.load("original.npy")
y1 = np.load("shifted.npy")

z = np.load("swin.npy")
z1 = np.load("swin1.npy")

#%% 
plt.imshow(y[0][0:196])
plt.show()

plt.imshow(y1[0][0:196])
plt.show()

print(np.linalg.norm(y))
print(np.linalg.norm(y1))


# pick two random tokens from same image y[0]
idx = np.random.randint(0,y.shape[1],2)
avg = np.mean(y[0][idx[0]]); avg1 = np.mean(y[0][idx[1]])
token = y[0][idx[0]].reshape(8,12) ; token1 = y[0][idx[1]].reshape(8,12)
print(avg, avg1)
token = np.where(token > avg , 255,  0); token1 = np.where(token1 > avg1, 255, 0)
plt.imshow(token)
plt.show()
plt.imshow(token1)
plt.show()
#%% 
# pick the same indexed token from both y and y1 
idx = np.random.randint(0,y.shape[1])
avg = np.mean(y[0][idx]); avg1 = np.mean(y1[0][idx])
token = y[0][idx].reshape(8,12); token1 = y1[0][idx].reshape(8,12)
print(avg, avg1)
token = np.where(token > avg , 255,  0); token1 = np.where(token1 > avg1, 255, 0)
plt.imshow(token)
plt.show()
plt.imshow(token1)
plt.show()

# TODO check original swin behavior as well 
# check content of each token and compare 

        


# %%



# # Find binary thresholded match
# y_mean = np.mean(y)
# y1_mean = np.mean(y1)
# y_binary =  np.where(y > y_mean, 255, 0)
# y1_binary = np.where(y1 > y1_mean, 255, 0)

# for i in tqdm(range(y.shape[1])):
#     anchor = y_binary[0][i]
#     if find_similar(anchor, y1_binary, 0.1):
#         print("found a match")
#         break
        


# %%


def find_similar(anchor:np.array, y_search:np.array, threshold:float):
    ''' Find matching of anchor in y_search matrix
    '''
    matches = []
    for i in range(y_search.shape[0]):
        if np.linalg.norm(anchor - y_search[i]) < threshold:
            matches.append(i)
    return matches

def find_shift(y, y1, early_break = False):
    shift_candidates = []
    for i in tqdm(range(y.shape[0])):
        anchor = y[i]
        matches = find_similar(anchor, y1, 0.1)
        if matches:
            assert len(matches) == 1
            if early_break:
                return i-matches[0]
            shift_dist = i-matches[0]
            if shift_dist < 0:
                shift_dist = y.shape[0] + shift_dist # account for wrapping index
            if shift_candidates and shift_dist != shift_candidates[0]:
                print("Warning: multiple shift candidates")
                print(f"Found shift candidate: {shift_dist}")
                print(f"Previous shift candidate: {shift_candidates[0]}")
                shift_candidates.append(shift_dist)
            elif not shift_candidates:
                shift_candidates.append(shift_dist)
                print(f"Found shift candidate: {shift_dist}")

    assert len(shift_candidates) > 0
    shift_size = shift_candidates[0]
    return shift_size

def shift_and_compare(y, y1, shift_size):

    y1_shifted = np.roll(y1, shift_size, axis=0)
    dist = np.linalg.norm(y1_shifted - y)
    return dist
# %%
def compare_tokens(y, y1):
    num_matches = 0
    num_total = 0
    for i in tqdm(range(y.shape[0])):
        y_token = y[i]
        y1_token = y1[i]
        if np.linalg.norm(y_token - y1_token) < 0.001:
            num_matches += 1
        num_total += 1
    
    return num_matches, num_total, num_matches/num_total
# %%

for img_id in [1,2]:
    print(f"Image {img_id}")
    y1_img = y1[img_id]
    y_img = y[img_id]

    shift_size = find_shift(y_img, y1_img, early_break=False)
    dist = shift_and_compare(y_img, y1_img, shift_size)
    print(f"shift size: {shift_size}")
    print(f"Distance using poly: {dist}")
    dist_orig = np.linalg.norm(y1_img - y_img)
    print(f"Distance w.o: {dist_orig}")





# %%