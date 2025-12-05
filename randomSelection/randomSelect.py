import os
import sys
import random
import shutil
from concurrent.futures import ThreadPoolExecutor

def copyFile(src, dest):
    shutil.copy(src, dest)

def select_files(source, target, percentage):
    # get all files
    allFiles = [f for f in os.listdir(source) if os.path.isfile(os.path.join(source, f))]
    
    # based off given percentage, calculate how many files will be selected
    numFiles = int(len(allFiles) * (percentage / 100))
    
    # shuffle and select
    selected = random.sample(allFiles, numFiles)
    
    # copy selected into new folder
    if not os.path.exists(target):
        os.makedirs(target)
    
    filePaths = [(os.path.join(source, file), os.path.join(target, file)) for file in selected]
    
    # copy files parallel
    with ThreadPoolExecutor() as executor:
        executor.map(lambda x: copyFile(*x), filePaths)
    
    print(f"{numFiles} files have been copied to {target}.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        percentage = int(sys.argv[1])

        sourceFolder = "/home/pnluong/CSE559Project/train_prefix"
        targetFolder = "/home/pnluong/CSE559Project/train_" + str(percentage)
        select_files(sourceFolder, targetFolder, percentage)