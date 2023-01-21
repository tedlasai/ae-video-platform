
import os
import time
import numpy as np
start_time = time.time()
#C:\Users\tedla\PycharmProjects\4d-data-browser\Image_Arrays_from_dng_separate
path ='C:/Users/tedla/PycharmProjects/4d-data-browser/Image_Arrays_exposure_separate/'
algoImages = []
listdir_ = os.listdir(path)
npy_frames = [path + f for f in listdir_ if f.endswith(('.npy'))]
print(npy_frames)

c = 0
for npy_frame in npy_frames:
  data = np.load(npy_frame)
  algoImages.append(data)
  print(c)
  c += 1

filename = 'Scene18_ds_raw_imgs'
save_loc = path[:-9]+'new'
os.makedirs(save_loc, exist_ok=True)
np.save(save_loc+'/'+filename, np.asarray(algoImages))
mid_time = time.time()
print("running time1: {:.3f}".format(mid_time-start_time))

path =  'C:/Users/tedla/PycharmProjects/4d-data-browser/Image_Arrays_from_dng_separate/'
showImages = []
listdir_ = os.listdir(path)
npy_frames = [path + f for f in listdir_ if f.endswith(('.npy'))]
c = 0

for npy_frame in npy_frames:
  data = np.load(npy_frame)
  showImages.append(data)
  print(c)
  c += 1

filename = 'Scene18_show_dng_imgs'
save_loc_ = path[:-10]
os.makedirs(save_loc_, exist_ok=True)
np.save(save_loc_+'/'+filename, np.asarray(showImages))
print("running time2: {:.3f}".format(time.time()-mid_time))