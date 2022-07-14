import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import cv2
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib as mp
import os
import glob
import platform

Image.MAX_IMAGE_PIXELS = None
path = "J:\Final\Temp_17"

joinPathChar = "/"
if(platform.system() == "Windows"):
    joinPathChar = "\\"

os.chdir(path)
my_files1 = glob.glob('*.JPG')
for i in range(len(my_files1)):
    check = os.path.abspath(my_files1[i])
    my_files1[i] = check

my_files1 = np.array(my_files1)
img_ar = []


for i in range(len(my_files1)):
    check = os.path.abspath(my_files1[i])
    # print(check)
    img_ar.append(cv2.imread(check))

print(img_ar)

# for i in range(len(my_files1)):
temp_stack = deepcopy(img_ar[0:15])

# Exposure fusion using Mertens
merge_mertens = cv2.createMergeMertens()
res_mertens = merge_mertens.process(temp_stack)
print(type(res_mertens))

# print(type(res_mertens))
# Convert datatype to 8-bit and save

res_mertens_8bit = np.clip(res_mertens * 255, 0, 255).astype('uint8')
cv2.putText(res_mertens_8bit, 'HDR-Mertens', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

img = Image.fromarray(res_mertens_8bit)
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
img.save("C:\\Users\\tedlasai\\PycharmProjects\\4d-data-browser\\HDR_Mertens_Video\\try.jpeg")