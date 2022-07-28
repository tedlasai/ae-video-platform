
from PIL import Image, ImageTk
import cv2
from copy import deepcopy
import os
import platform


def main(scene, fps, exposure, exp_list, folder):

    Image.MAX_IMAGE_PIXELS = None

    joinPathChar = "/"
    if(platform.system() == "Windows"):
        joinPathChar = "\\"

    save_loc = os.path.join(os.path.dirname(__file__), 'Regular_Videos')
    os.makedirs(save_loc, exist_ok=True)

    my_fold = os.listdir(folder)
    filtered_path = []
    scene_name = []

    for i in my_fold:
        if "Scene" in i:

            loc = folder + joinPathChar + i
            filtered_path.append(loc)
            if (platform.system() == "Windows"):
                scene_name.append((loc.split("_")[0]).split("\\")[2])
            else:
                scene_name.append((loc.split("_")[0]).split("/")[2])

    index = scene_name.index(scene)
    print(filtered_path[index])
    path = filtered_path[index]

    my_files1 = [path + joinPathChar + f for f in os.listdir(path + joinPathChar) if f.endswith(('.jpg','.JPG', '.jpeg','.JPEG', '.png', '.PNG'))]
    img_ar = []

    for i in range(100):

        print("i is ", i)
        temp_img_ind = int(i * 15 + exposure)
        check = os.path.abspath(my_files1[temp_img_ind])
        print(check)
        im = cv2.imread(check)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

        img_ar.append(im)

    img = Image.fromarray(img_ar[0])
    vid_name = save_loc + joinPathChar + str(scene) + "_1.0_Ex_" + exp_list[exposure] + "_FPS_" + str(fps) + ".avi"
    video = cv2.VideoWriter(vid_name, cv2.VideoWriter_fourcc('M', 'J', "P", 'G'), fps, (img.width, img.height))

    for i in range(len(img_ar)):

        img = img_ar[i]
        print("video product ", i)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        video.write(img)

