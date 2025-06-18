import os
import random

path_to_test = "/dlocal/home/2022022/tmayet02/datasets/places/test_256"

# list files names in path
files_names = os.listdir(path_to_test)

# sort files names
files_names = sorted(files_names)

# keep .jpg
files_names = [file_name for file_name in files_names if file_name.endswith(".jpg")]

# samples randomly 2000 images

random.seed(42)
random.shuffle(files_names)

with open("places_2000_filesnames_test.txt", "w") as f:
    for file_name in files_names[:2000]:
        f.write(f"{file_name}\n")
