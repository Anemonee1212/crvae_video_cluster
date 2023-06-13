import io
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import shutil

# Chinese videos
source_dir = "images/cn/"
target_dir = "input_data/cn/"

with io.open("input_data/cn.txt", "r", encoding = "UTF-8") as f:
    text = f.readlines()
    print(len(text))

print(os.listdir(source_dir))

target_idx = 0
for i in range(1, 379, 21):
    for j in range(0, 20, 4):
        source_file_name = str(i + j) + ".jpg"
        target_file_name = str(target_idx) + ".jpg"
        shutil.copyfile(src = source_dir + source_file_name, dst = target_dir + target_file_name)
        target_idx += 1

print(os.listdir(target_dir))

# United States videos
source_dir = "images/us/"
target_dir = "input_data/us/"
print(os.listdir(source_dir))

data = pd.read_csv("input_data/us.txt", index_col = 0)
print(data)

plt.plot(data.start)
plt.show()

image_indices = np.rint(data.start / 2) + 1
print(image_indices)

for i, num in enumerate(image_indices):
    if num <= 109:
        source_file_name = str(int(num)) + ".jpg"
        target_file_name = str(i) + ".jpg"
        shutil.copyfile(src = source_dir + source_file_name, dst = target_dir + target_file_name)

shutil.copyfile(source_dir + "108.jpg", target_dir + str(len(data) - 1) + ".jpg")
print(os.listdir(target_dir))
