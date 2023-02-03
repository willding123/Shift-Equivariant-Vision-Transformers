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
# dir_path = "/fs/nexus-scratch/pding/output/swin_tiny_patch4_window7_1k_poly/default"
dir_path = "/fs/nexus-projects/shift-equivariant_vision_transformer/swin_tiny_patch4_window7_224_22k/default"

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
plt.show()

# %%
