#%%
import os
import re
import matplotlib.pyplot as plt

def process(dir_path, file_name, acc_ls):
    with open(os.path.join(dir_path, file_name), "r") as file:
            # iterate through each line in the file
            for line in file:
            # use a regular expression to tokenize the line
                if "INFO  * Acc@1" in line: 
                    #INFO Max accuracy: 2.90%
                    match  = re.search(r"\d+\.\d+", line)
                    if match:
                        acc_ls.append(float(match.group(0))/100)


acc_ls = []
waitlist = []
# directory where the log_rank files are located
# dir_path = "/fs/nexus-scratch/pding/output/swin_tiny_patch4_window7_1k_default/default"
# dir_path = "/fs/nexus-projects/shift-equivariant_vision_transformer/poly_swin_tiny_0215/default"
dir_path = "/home/pding/scratch.cmsc663/pvit_noaug405/default"
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

fig, ax = plt.subplots()
ax.plot(acc_ls)
# plt.show()

# display the plot using log base 2 scale for both x and y axis
ax.set_xscale('log', basex=10)
ax.plot(acc_ls)
plt.show()
#%%
# write a power law function
def power_law(x, a, b):
    return a * x ** b

# fit the data using curve_fit
from scipy.optimize import curve_fit
popt, pcov = curve_fit(power_law, range(1, len(acc_ls)+1), acc_ls)

# plot the data and the fitted function
fig, ax = plt.subplots()
ax.plot(range(1, len(acc_ls)+1), acc_ls, 'o', label='data')
ax.plot(range(1, len(acc_ls)+1), power_law(range(1, len(acc_ls)+1), *popt), 'r-', label='fit')
ax.set_xscale('log', basex=10)
ax.set_yscale('log', basey=10)
ax.legend()

# display the plot
plt.show()

# plot the extrapolated data
fig, ax = plt.subplots()
ax.plot(range(1, len(acc_ls)+1), acc_ls, 'o', label='data')
ax.plot(range(1, len(acc_ls)+1), power_law(range(1, len(acc_ls)+1), *popt), 'r-', label='fit')
ax.plot(range(1, 301), power_law(range(1, 301), *popt), 'g-', label='extrapolation')
ax.set_xscale('log', basex=10)
ax.set_yscale('log', basey=10)
ax.legend()
plt.show()

# extrapolate the data to 300 epochs: "extrapolate to 300 epochs:"
print("extrapolate to 300 epochs: ", power_law(300, *popt))

# %%
