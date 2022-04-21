import numpy as np
import os
import cv2

read_path='C:/Users/tedlasai/OneDrive - York University/School/York/Lab/ExposureData/latest/'
dir_count=0
scene_num='3'
downscale_ratio=0.12

list_of_images=[]
list_of_img_mean=[]

images = [read_path + '/'+ f for f in os.listdir(read_path + '/') if f.endswith(('.jpg','.JPG', '.jpeg','.JPEG', '.png', '.PNG'))]
images.sort()
count=0
for image_name in images:
    print(count)
    count+=1
    temp_image=cv2.imread(image_name,-1)
    temp_image_resized=cv2.resize(temp_image,None,fx=downscale_ratio,fy=downscale_ratio)[:,:,::-1]
    list_of_images.append(temp_image_resized)
    list_of_img_mean.append(np.mean(temp_image_resized))

np.save('Scene'+scene_num+'_imgs_'+str(downscale_ratio),np.asarray(list_of_images))
np.save('Scene'+scene_num+'_img_mean_'+str(downscale_ratio),np.asarray(list_of_img_mean))