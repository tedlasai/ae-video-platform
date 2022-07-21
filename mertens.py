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

def main(scene, fps):

    Image.MAX_IMAGE_PIXELS = None
    # path = "J:\Final\Scene9_WindowDiffuse"

    joinPathChar = "/"
    if(platform.system() == "Windows"):
        joinPathChar = "\\"

    folder = "J:\Final"


    my_fold = os.listdir(folder)
    filtered_path = []
    scene_name = []

    for i in my_fold:
        if "Scene" in i:

            loc = "J:\Final" + "\\" + i
            filtered_path.append(loc)
            scene_name.append((loc.split("_")[0]).split("\\")[2])

    # print(filtered_path)
    index = scene_name.index(scene)
    # print(index)
    print(filtered_path[index])
    path = filtered_path[index]

    os.chdir(path)
    my_files1 = glob.glob('*.JPG')

    mertens_ar = np.load(os.path.join(os.path.dirname(__file__), 'Image_Arrays') + "\\" + scene + '_mertens_imgs_' + str(1.0) + '.npy')

    # for i in range(100):
    #
    #
    #     print("i is ", i)
    #
    #     temp_img_ind = int(i * 15)
    #     # temp_stack = deepcopy(my_files1[temp_img_ind:temp_img_ind + 15])
    #     img_ar = []
    #
    #     for j in range(15):
    #
    #         print("j is ", j)
    #         check = os.path.abspath(my_files1[temp_img_ind+j])
    #         im = cv2.imread(check)
    #         im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    #         img_ar.append(im)
    #
    #     temp_stack = deepcopy(img_ar[0:15])
    #
    #     # Exposure fusion using Mertens
    #     merge_mertens = cv2.createMergeMertens()
    #     res_mertens = merge_mertens.process(temp_stack)
    #     print(type(res_mertens))
    #
    #     # print(type(res_mertens))
    #     # Convert datatype to 8-bit and save
    #
    #     res_mertens_8bit = np.clip(res_mertens * 255, 0, 255).astype('uint8')
    #     cv2.putText(res_mertens_8bit, 'HDR-Mertens', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    #
    #     mertens_ar.append(res_mertens_8bit)


    # print(len(mertens_ar))

    img = Image.fromarray(mertens_ar[0])

    vid_name = os.path.join(os.path.dirname(__file__), 'HDR_Mertens_Video') + "\\" + str(scene) + "_1.0_Mertens_" + "_FPS_" + str(fps) + ".avi"

    video = cv2.VideoWriter(vid_name , cv2.VideoWriter_fourcc('M', 'J', "P", 'G'), fps, (img.width, img.height))

    for i in range(len(mertens_ar)):

        img = mertens_ar[i]
        print("video product ", i)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        video.write(img)



# main("Scene13", 10)