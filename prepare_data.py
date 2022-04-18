import numpy as np
import os
import cv2

read_path='D:/DCIM/100EOS5D/'
dir_count=0

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

np.save('Scene2_imgs_'+str(downscale_ratio),np.asarray(list_of_images))
np.save('Scene2_img_mean_'+str(downscale_ratio),np.asarray(list_of_img_mean))