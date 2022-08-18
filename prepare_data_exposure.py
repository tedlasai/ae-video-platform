import os
import numpy as np
import rawpy as rawpy
#import exifread
import time

# def getMata_exifread(filePath):
#     f = open(filePath, "rb")
#     exif_0 = exifread.process_file(f)
#     return exif_0['EXIF ExposureTime'], exif_0['EXIF ISOSpeedRatings']


# https://stackoverflow.com/questions/1806278/convert-fraction-to-float
def fraction_str_to_float(str):
    try:
        return float(str)
    except ValueError:
        try:
            num, denom = str.split('/')
        except ValueError:
            return None
        try:
            leading, num = num.split(' ')
        except ValueError:
            return float(num) / float(denom)
        if float(leading) < 0:
            sign_mult = -1
        else:
            sign_mult = 1
        return float(leading) + sign_mult * (float(num) / float(denom))

start_time = time.time()
NUMBER_OF_IMAGES_PER_STACK = 15
read_path = 'D:/Final/Scene20_MeteorDebris/'
scene_num = '20'
save_loc = os.path.join(os.path.dirname(__file__), 'Image_Arrays_exposure')
os.makedirs(save_loc, exist_ok=True)
joinPathChar = "/"

listdir_ = os.listdir(read_path)
images = [read_path + f for f in listdir_ if f.endswith(('.cr2', '.CR2'))]

list_of_ims = []
list_of_isos = []
list_of_shutter_speeds = []

if (len(images) % NUMBER_OF_IMAGES_PER_STACK) != 0:
    print("Wrong number of input images or Wrong number of images per stack!")
else:
    number_of_stacks = int(len(images) / NUMBER_OF_IMAGES_PER_STACK)
    images.sort()
    one_stack_ims_temp_list = []
    one_stack_isos_temp_list = []
    one_stack_shutter_speeds_temp_list = []
    for im_path in images:
        raw_im = rawpy.imread(im_path)
        raw_bayer = raw_im.raw_image.copy()



        #scale data between 0-255
        #this is the active area for this sensor and also takes into account the default crop
        raw_bayer = raw_bayer [54:4544-10,148:6868]
        #print(np.right_shift(np.array([16383], 6)))

        raw_bayer_downscaled = np.right_shift(raw_bayer, 6)
        raw_bayer_downscaled = raw_bayer_downscaled[::4,::4]

        raw_bayer_downscaled = raw_bayer_downscaled.astype(np.uint8)
        one_stack_ims_temp_list.append(raw_bayer_downscaled)
        # shutter_speed,iso = getMata_exifread(im_path)
        # one_stack_isos_temp_list.append(int(str(iso)))
        # ex_time = fraction_str_to_float(str(shutter_speed))
        # one_stack_shutter_speeds_temp_list.append(round(ex_time,3))
        if len(one_stack_ims_temp_list) == NUMBER_OF_IMAGES_PER_STACK:
            list_of_ims.append(one_stack_ims_temp_list)
            # list_of_isos.append(one_stack_isos_temp_list)
            # list_of_shutter_speeds.append(one_stack_shutter_speeds_temp_list)
            one_stack_ims_temp_list = []
            # one_stack_isos_temp_list = []
            # one_stack_shutter_speeds_temp_list = []

    np.save(save_loc + joinPathChar + 'Scene' + scene_num + '_ds_raw_imgs',np.asarray(list_of_ims))
    # np.save('Scene_img_ex_times',np.asarray(list_of_shutter_speeds))
    # np.save('Scene_img_isos',np.asarray(list_of_isos))

print("running time: {:.3f}".format(time.time()-start_time))