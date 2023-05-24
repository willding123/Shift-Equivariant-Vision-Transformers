#%%
import os

root_dir = os.path.expanduser("~/scratch.cmsc663/Swin-Transformer/wandb")

for root, dirs, files in os.walk(root_dir):
    if "files" in dirs:
        files_dir = os.path.join(root, "files")
        log_files = [f for f in os.listdir(files_dir) if f.endswith(".log")]
        for log_file in log_files:
            log_file_path = os.path.join(files_dir, log_file)
            with open(log_file_path) as f:
                if "INFO Train" in f.read():
                    print(root)
                    break
#%%