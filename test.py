import os
import random
import shutil

source_folder = r'./DATASET/TRAIN/R'
target_folder = r'./DATASET/VAL/R'

files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]


num_files_to_copy = int(len(files) * 0.2)
selected_files = random.sample(files, num_files_to_copy)


for file in selected_files:
    src = os.path.join(source_folder, file)
    dst = os.path.join(target_folder, file)
    shutil.copy2(src, dst)

print(f"Đã copy {len(selected_files)} file.")