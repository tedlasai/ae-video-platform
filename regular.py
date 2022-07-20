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


def main(scene, fps, exposure, exp_list):

    Image.MAX_IMAGE_PIXELS = None

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

    index = scene_name.index(scene)
    print(index)
    print(filtered_path[index])

    path = "J:\Final\Scene11_MovingHeadBacklight"
    path = filtered_path[index]

    os.chdir(path)
    my_files1 = glob.glob('*.JPG')
    mertens_ar = []
    img_ar = []

    for i in range(100):

        print("i is ", i)
        temp_img_ind = int(i * 15 + exposure)
        check = os.path.abspath(my_files1[temp_img_ind])
        im = cv2.imread(check)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        temp_stack = deepcopy(im)
        img_ar.append(im)

    img = Image.fromarray(img_ar[0])

    vid_name = "C:\\Users\\tedlasai\\PycharmProjects\\4d-data-browser\\Regular_Videos\\" + str(scene) + "_1.0_Ex_" + exp_list[exposure] + "_FPS_" + str(fps) + ".avi"

    video = cv2.VideoWriter(vid_name, cv2.VideoWriter_fourcc('M', 'J', "P", 'G'), fps, (img.width, img.height))

    for i in range(len(img_ar)):

        img = img_ar[i]
        print("video product ", i)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        video.write(img)




# main("Scene1", 10, 13, ['15', '8', '6', '4', '2', '1', '05', '1-4', '1-8', '1-15', '1-30', '1-60', '1-125', '1-250', '1-500'])