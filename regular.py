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


def main():

    Image.MAX_IMAGE_PIXELS = None
    path = "J:\Final\Scene11_MovingHeadBacklight"

    joinPathChar = "/"
    if(platform.system() == "Windows"):
        joinPathChar = "\\"

    folder = "J:\Final"

    my_fold = os.listdir(folder)
    filtered_path = []

    # for i in my_fold:
    #     if "Scene" in i:
    #
    #         loc = "J:\Final" + "\\" + i
    #         filtered_path.append(loc)
    #
    # print(filtered_path)


    os.chdir(path)
    my_files1 = glob.glob('*.JPG')


    mertens_ar = []
    img_ar = []

    for i in range(100):


        print("i is ", i)

        temp_img_ind = int(i * 15 + 8)

        check = os.path.abspath(my_files1[temp_img_ind])
        im = cv2.imread(check)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

        temp_stack = deepcopy(im)

        img_ar.append(im)



    img = Image.fromarray(img_ar[0])

    video = cv2.VideoWriter("C:\\Users\\tedlasai\\PycharmProjects\\4d-data-browser\\Regular_Videos\\Scene11_1.0_Ex_1-8_FPS_10.avi",
                            cv2.VideoWriter_fourcc('M', 'J', "P", 'G'), 10, (img.width, img.height))

    for i in range(len(img_ar)):

        img = img_ar[i]
        print("video product ", i)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        video.write(img)
        # cv2.imwrite("C:\\Users\\tedlasai\\PycharmProjects\\4d-data-browser\\HDR_Mertens_Video\\try.jpeg", img)
        #img.save("C:\\Users\\tedlasai\\PycharmProjects\\4d-data-browser\\HDR_Mertens_Video\\try.jpeg")

