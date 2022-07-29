from PIL import Image, ImageTk
import numpy as np
import cv2
from copy import deepcopy
import os
import platform

def main():

    joinPathChar = "/"
    if (platform.system() == "Windows"):
        joinPathChar = "\\"

    Image.MAX_IMAGE_PIXELS = None

    read_path = 'F:/DCIM/104EOS5D'

    save_loc = os.path.join(os.path.dirname(__file__), 'Image_Arrays')
    os.makedirs(save_loc, exist_ok=True)

    scene_num= '20'
    downscale_ratio_low = 0.12
    downscale_ratio_high = 1.00

    list_of_images=[]
    list_of_img_mean=[]
    mertens_ar_low = []
    mertens_ar_high = []
    img_ar_low = []
    img_ar_high = []

    images = [read_path + "/" + f for f in os.listdir(read_path + '/') if f.endswith(('.jpg','.JPG', '.jpeg','.JPEG', '.png', '.PNG'))]
    images.sort()
    count= 0

    # regular images into npy

    for image_name in images:
        print(count)
        count+=1
        temp_image=cv2.imread(image_name,-1)
        temp_image_resized=cv2.resize(temp_image,None,fx=downscale_ratio_low,fy=downscale_ratio_low)[:,:,::-1]
        list_of_images.append(temp_image_resized)
        list_of_img_mean.append(np.mean(temp_image_resized))

    #  mertens high and low resolution

    for i in range(100):

        temp_img_ind = int(i * 15)

        for j in range(15):
            check = os.path.abspath(images[temp_img_ind + j])
            print(check, "i is ", i, "j is ", j)
            im = cv2.imread(check)
            im_low = cv2.resize(im, None, fx=downscale_ratio_low, fy=downscale_ratio_low)[:, :, ::-1]
            im_high = cv2.resize(im, None, fx=downscale_ratio_high, fy=downscale_ratio_high)[:, :, ::-1]

            # im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            img_ar_low.append(im_low)
            img_ar_high.append(im_high)

        temp_stack_low = deepcopy(img_ar_low[0:15])
        temp_stack_high = deepcopy(img_ar_high[0:15])

        mertens_ar_low.append(mertens_process(temp_stack_low))
        mertens_ar_high.append(mertens_process(temp_stack_high))

    np.save(
        save_loc + joinPathChar + 'Scene' + scene_num + '_imgs_' + str(
            downscale_ratio_low), np.asarray(list_of_images))
    np.save(
        save_loc + joinPathChar + 'Scene' + scene_num + '_img_mean_' + str(
            downscale_ratio_low), np.asarray(list_of_img_mean))

    np.save(
        save_loc + joinPathChar + "Scene" + scene_num + '_mertens_imgs_' + str(
            downscale_ratio_high), np.asarray(mertens_ar_high))

    np.save(
        save_loc + joinPathChar + "Scene" + scene_num + '_mertens_imgs_' + str(
            downscale_ratio_low), np.asarray(mertens_ar_low))

def mertens_process(res_img):


    # Exposure fusion using Mertens

    merge_mertens = cv2.createMergeMertens()
    res_mertens = merge_mertens.process(res_img)

    # Convert datatype to 8-bit and save

    res_mertens_8bit = np.clip(res_mertens * 255, 0, 255).astype('uint8')
    cv2.putText(res_mertens_8bit, 'HDR-Mertens', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    return res_mertens_8bit


