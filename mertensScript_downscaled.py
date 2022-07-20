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
path = "J:\Final\Scene9_WindowDiffuse"

joinPathChar = "/"
if(platform.system() == "Windows"):
    joinPathChar = "\\"

folder = "J:\Final"

downscale_ratio=0.12
my_fold = os.listdir(folder)
filtered_path = []

for i in my_fold:
    if "Scene" in i:

        loc = "J:\Final" + "\\" + i
        filtered_path.append(loc)

for i in range(len(filtered_path)):
    print(i, filtered_path[i])
count = 0

for loc_path in range(1): #, len(filtered_path)

    print(filtered_path[loc_path])
    os.chdir(filtered_path[loc_path])
    my_files1 = glob.glob('*.JPG')

    mertens_ar = []
    img_ar = []

    print("mertens length 1 is ", len(mertens_ar))

    for i in range(100):

        print("i is ", i)

        temp_img_ind = int(i * 15)
        # temp_stack = deepcopy(my_files1[temp_img_ind:temp_img_ind + 15])
        img_ar = []

        for j in range(15):

            print("j is ", j)
            check = os.path.abspath(my_files1[temp_img_ind+j])
            print(check, "i is ", i, "j is ", j, filtered_path[loc_path])
            im = cv2.imread(check)
            im2 = cv2.resize(im, None, fx=downscale_ratio, fy=downscale_ratio)[:, :, ::-1]
            # im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            img_ar.append(im2)

        temp_stack = deepcopy(img_ar[0:15])

        # Exposure fusion using Mertens
        merge_mertens = cv2.createMergeMertens()
        res_mertens = merge_mertens.process(temp_stack)
        print(type(res_mertens))

        # print(type(res_mertens))
        # Convert datatype to 8-bit and save

        res_mertens_8bit = np.clip(res_mertens * 255, 0, 255).astype('uint8')
        cv2.putText(res_mertens_8bit, 'HDR-Mertens', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        mertens_ar.append(res_mertens_8bit)


    print("mertens length 2 is ", len(mertens_ar))

    np.save('C:\\Users\\tedlasai\\PycharmProjects\\4d-data-browser\\' + (filtered_path[loc_path].split("_")[0]).split("\\")[2] + '_mertens_imgs_' + str(downscale_ratio), np.asarray(mertens_ar))

    mertens_ar = []
    # np.save('Scene' + scene_num + '_img_mean_' + str(downscale_ratio), np.asarray(list_of_img_mean))

    # img = Image.fromarray(mertens_ar[0])
    #
    # upload_location = "C:\\Users\\tedlasai\\PycharmProjects\\4d-data-browser\\HDR_Mertens_Video\\" + (filtered_path[loc_path].split("_")[0]).split("\\")[2] + "_1.0_Mertens_FPS_10.avi"
    #
    # video = cv2.VideoWriter(upload_location, cv2.VideoWriter_fourcc('M', 'J', "P", 'G'), 10,
    #                                 (img.width, img.height))
    #
    # for i in range(len(mertens_ar)):
    #
    #     img = mertens_ar[i]
    #     print("video product ", i)
    #
    #     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    #     video.write(img)
    #     # cv2.imwrite("C:\\Users\\tedlasai\\PycharmProjects\\4d-data-browser\\HDR_Mertens_Video\\try.jpeg", img)
    #     #img.save("C:\\Users\\tedlasai\\PycharmProjects\\4d-data-browser\\HDR_Mertens_Video\\try.jpeg")
