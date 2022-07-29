
from PIL import Image
import numpy as np
import cv2
import os
import platform

def main(scene, fps, folder):

    Image.MAX_IMAGE_PIXELS = None
    # path = "J:\Final\Scene9_WindowDiffuse"

    joinPathChar = "/"
    if(platform.system() == "Windows"):
        joinPathChar = "\\"

    save_loc = os.path.join(os.path.dirname(__file__), 'HDR_Mertens_Video')
    os.makedirs(save_loc, exist_ok=True)


    my_fold = os.listdir(folder)
    filtered_path = []
    scene_name = []

    for i in my_fold:
        if "Scene" in i:

            loc = folder + joinPathChar + i
            filtered_path.append(loc)
            scene_name.append((loc.split("_")[0]).split(joinPathChar)[2])


    print(filtered_path[scene_name.index(scene)])
    path = filtered_path[scene_name.index(scene)]

    mertens_ar = np.load(os.path.join(os.path.dirname(__file__), 'Image_Arrays') + joinPathChar + scene + '_mertens_imgs_' + str(1.0) + '.npy')

    img = Image.fromarray(mertens_ar[0])

    vid_name = save_loc + joinPathChar + str(scene) + "_1.0_Mertens_" + "_FPS_" + str(fps) + ".avi"

    video = cv2.VideoWriter(vid_name , cv2.VideoWriter_fourcc('M', 'J', "P", 'G'), fps, (img.width, img.height))

    for i in range(len(mertens_ar)):

        img = mertens_ar[i]
        print("video product ", i)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        video.write(img)

