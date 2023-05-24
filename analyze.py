#%%
import os
import re
import matplotlib.pyplot as plt

def process(dir_path, file_name, acc_ls):
    with open(os.path.join(dir_path, file_name), "r") as file:
            # iterate through each line in the file
            for line in file:
            # use a regular expression to tokenize the line
                if "Accuracy of the network on the" in line: 
                    #INFO Max accuracy: 2.90%
                    match  = re.search(r"\d+\.\d+", line)
                    if match:
                        acc_ls.append(float(match.group(0))/100)


acc_ls = []
waitlist = []
# directory where the log_rank files are located
# dir_path = "/fs/nexus-scratch/pding/output/swin_tiny_patch4_window7_1k_default/default"
# dir_path = "/fs/nexus-projects/shift-equivariant_vision_transformer/poly_swin_tiny_0215/default"
dir_path = "/home/pding/scratch.cmsc663/ptwins_svts_b_scratch/default"
# dir_path = "/scratch/zt1/project/furongh-prj/shared/pvit_b_1kscratch/default"
i = 0
# iterate through the files in the directory
for file_name in os.listdir(dir_path):
  # check if the file starts with "log_rank"
  if file_name.startswith("log_rank0"):
    if str(i) in file_name: 
        process(dir_path, file_name, acc_ls)     
        i += 1
        tmp = [f for f in waitlist if str(i) in f]
        while  tmp:
            process(dir_path, tmp[0], acc_ls)
            i += 1
            waitlist.pop(0)
            if waitlist: 
                tmp = [f for f in waitlist if str(i) in f]

    else: 
        waitlist.append(file_name)

#%%
# write a power law function
# write a power law function with three parameters a, b, and c
def power_law(x, a, b, c):
    return a * x ** b + c

# fit the data using curve_fit
from scipy.optimize import curve_fit
if len(acc_ls) < 50: 
    popt, pcov = curve_fit(power_law, range(1, len(acc_ls)+1), acc_ls, maxfev=5000)
else:
    popt, pcov = curve_fit(power_law, range(51, len(acc_ls)+1), acc_ls[50:], maxfev=10000)

# plot the data and the fitted function
step = 1
fig, ax = plt.subplots()
ax.plot(range(1, len(acc_ls)+1, step), acc_ls, 'o', label='data')
ax.plot(range(1, len(acc_ls)+1, step), power_law(range(1, len(acc_ls)+1), *popt), 'r-', label='fit')
# ax.set_xscale('log', basex=10)
# ax.set_yscale('log', basey=10)
ax.legend()
# add names to the x and y axis
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy')
# add a title to the plot
ax.set_title('Accuracy vs Epoch')
# print name of the model
print("Model: ", dir_path.split("/")[-2]) 
# print max accuracy and the epoch it occurs and description
print("Max accuracy: ", max(acc_ls), "at epoch: ", acc_ls.index(max(acc_ls)))
print("Last accuracy: ", acc_ls[-1], "at epoch: ", len(acc_ls)-1)
# display the plot
plt.show()

# plot the extrapolated data
extra = 50
fig, ax = plt.subplots()
ax.plot(range(1, len(acc_ls)+1, step), acc_ls, 'o', label='data')
ax.plot(range(1, len(acc_ls)+1, step), power_law(range(1, len(acc_ls)+1), *popt), 'r-', label='fit')
ax.plot(range(1, len(acc_ls)+50, step), power_law(range(1, len(acc_ls)+extra), *popt), 'g-', label='extrapolation')
# ax.set_xscale('log', basex=10)
# ax.set_yscale('log', basey=10)
# add names to the x and y axis
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy')
# add a title to the plot
ax.set_title('Extrapolated Accuracy vs Epoch')
ax.legend()
plt.show()

# extrapolate the data to n epochs: "extrapolate to n epochs:"
n = 300
print(f"extrapolate to {extra} epochs: ", power_law(n, *popt))
print(f"extrapolate to {n} epochs: ", power_law(n, *popt))

# %%
