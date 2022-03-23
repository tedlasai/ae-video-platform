import numpy as np
import os
import cv2

read_path='C:/Users/abd9r/OneDrive - York University/ColorfulScene/'
all_dir=[ _dir for _dir in os.listdir(read_path) if os.path.isdir(os.path.join(read_path, _dir)) ]
all_dir.sort()
dir_count=0

downscale_ratio=0.12

list_of_images=[]
list_of_ex_time=[]
list_of_img_mean=[]

for _dir in all_dir:
    dir_count+=1
    images = [read_path + _dir + '/'+ f for f in os.listdir(read_path + _dir + '/') if f.endswith(('.jpg','.JPG', '.jpeg','.JPEG', '.png', '.PNG'))]
    # images.sort()
    print(dir_count)
    for image_name in images:
        exposure_value=float((image_name.split('.jpg')[0]).split('_')[-1])
        if dir_count == 1:
            list_of_ex_time.append(exposure_value)
        temp_image=cv2.imread(image_name,-1)
        temp_image_resized=cv2.resize(temp_image,None,fx=downscale_ratio,fy=downscale_ratio)[:,:,::-1]
        list_of_images.append(temp_image_resized)
        list_of_img_mean.append(np.mean(temp_image_resized))

np.save('Scene1_imgs',np.asarray(list_of_images))
np.save('Scene1_ex_times',np.asarray(list_of_ex_time))
np.save('Scene1_img_mean',np.asarray(list_of_img_mean))