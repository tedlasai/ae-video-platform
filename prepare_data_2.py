import numpy as np
import os
import cv2

read_path='C:/Users/abd9r/OneDrive - York University/ColorfulScene/'

downscale_ratio=0.12

list_of_img=[]
list_of_img_mean=[]

ex_times=np.asarray(np.load('Scene1_ex_times.npy'))
ex_times.sort()

frame_num=90

for i in range(1,frame_num+1):
    print('Frame #: ', i)
    for j in range(len(ex_times)):
        image_name = read_path+'Frame_'+str(i)+'/'+'Frame_'+str(i)+'_Shutter_Value_'+str(ex_times[j])+'.jpg'
        temp_image=cv2.imread(image_name,-1)
        temp_image_resized=cv2.resize(temp_image,None,fx=downscale_ratio,fy=downscale_ratio)[:,:,::-1]
        list_of_img.append(temp_image_resized)
        list_of_img_mean.append(np.mean(temp_image_resized))

np.save('Scene1_imgs',np.asarray(list_of_img))
np.save('Scene1_img_mean',np.asarray(list_of_img_mean))