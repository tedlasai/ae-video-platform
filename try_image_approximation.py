import os
import time

import cv2
import numpy as np
import rawpy
from simple_camera_pipeline.python.pipeline import run_pipeline_v2,get_metadata

SCALE_LABELS = [15,8,6,4,2,1,1/2,1/4,1/8,1/15,1/30,1/60,1/125,1/250,1/500]
NEW_SCALES = [15,13,10,8,6,5,4,3.2,2.5,2,1.6,1.3,1,0.8,0.6,0.5,0.4,0.3,1/4,1/5,1/6,1/8,1/10,1/13,1/15,1/20,1/25,1/30,1/40,1/50,1/60,1/80,1/100,1/125,1/160,1/200,1/250,1/320,1/400,1/500]
means = []
NUMBER_OF_IMAGES_PER_STACK = len(SCALE_LABELS)
NUMBER_OF_IMAGES_PER_STACK_NEW = len(NEW_SCALES)
TOTAL_IMS = 100 * NUMBER_OF_IMAGES_PER_STACK_NEW

#image 3 4", 4 2", 5 1", approximate 4 with 3 and 5,
def one_pixel_function(im1,im2,x1,x2,targetx):
    result = (im1/x1 + im2/x2)*targetx/2
    result[(im1 == 1) & (im2 < 1)] = im2[(im1 == 1) & (im2 < 1)] * targetx / x2
    result[(im1 > 0) & (im2 == 0)] = im1[(im1 > 0) & (im2 == 0)] * targetx / x1
    result[im2 == 1] = 1
    result[result > 1] = 1
    return result

# does not assue line across origin
def one_pixel_function_2(im1,im2,x1,x2,targetx):
    result = (im1-im2)/(x1-x2)*(targetx-x1)+im1
    result[(im1 == 1) & (im2 < 1)] = im2[(im1 == 1) & (im2 < 1)] * targetx / x2
    result[(im1 > 0) & (im2 == 0)] = im1[(im1 > 0) & (im2 == 0)] * targetx / x1
    result[im2 == 1] = 1
    result[result > 1] = 1
    return result


def get_normed_im(image_path):
    raw_im = rawpy.imread(image_path)
    raw_bayer = raw_im.raw_image.copy()
    output_im_algorithm,output_im_show=save_im(raw_bayer, image_path)
    raw_bayer = raw_bayer/(2**14-1)
    np.clip(raw_bayer, 0, 1, out=raw_bayer)
    mean = np.mean(raw_bayer)
    return raw_bayer,mean,output_im_algorithm,output_im_show


def method_1_assume_line_across_origin(image_path1,image_path2,ind1,ind2,targetx):
    im1,mean1 = get_normed_im(image_path1)
    im2,mean2 = get_normed_im(image_path2)
    x1 = SCALE_LABELS[ind1]
    x2 = SCALE_LABELS[
        2]
    return one_pixel_function(im1,im2,x1,x2,targetx)

def method_2(image_path1,image_path2,ind1,ind2,targetx):
    im1,mean1 = get_normed_im(image_path1)
    im2,mean2 = get_normed_im(image_path2)
    x1 = SCALE_LABELS[ind1]
    x2 = SCALE_LABELS[ind2]
    return one_pixel_function_2(im1,im2,x1,x2,targetx)

def method(im1,im2,ind1,ind2,targetx):
    x1 = SCALE_LABELS[ind1]
    x2 = SCALE_LABELS[ind2]
    return one_pixel_function_2(im1,im2,x1,x2,targetx)

# def compare_approxi_caputured(im_path,normed_im):
#     captured,mean = get_normed_im(im_path)
#     diff = np.abs(captured-normed_im)
#     m = np.max(diff)
#     min = np.min(diff)
#     diff_sum = np.sum(diff)
#     captured = captured*(2**14-1)
#     save_im(captured,im_path)
#     return diff_sum

def make_npy_data_for_algorithem_one_im(c1,c2,c3,c4,shape0,shape1):
    c1_ = c1[::2, ::2]
    c2_ = c2[1::2, ::2]
    c3_ = c3[::2, 1::2]
    c4_ = c4[1::2, 1::2]
    raw_bayer_downscaled = np.empty((int(shape0 * 0.25), int(shape1 * 0.25)),dtype=np.uint16)
    raw_bayer_downscaled[::2, ::2] = c1_
    raw_bayer_downscaled[1::2, ::2] = c2_
    raw_bayer_downscaled[::2, 1::2] = c3_
    raw_bayer_downscaled[1::2, 1::2] = c4_
    #raw_bayer_downscaled = np.right_shift(raw_bayer_downscaled, 6)
    black_level = 512
    white_level = 14008
    raw_bayer_downscaled = np.clip(raw_bayer_downscaled,black_level,white_level)
    raw_bayer_downscaled = ((raw_bayer_downscaled-black_level)/(white_level - black_level)) * 255
    raw_bayer_downscaled = raw_bayer_downscaled.astype(np.uint8)
    return raw_bayer_downscaled


    # one_stack_ims_temp_list.append(raw_bayer_downscaled)
    # #         # shutter_speed,iso = getMata_exifread(im_path)
    # #         # one_stack_isos_temp_list.append(int(str(iso)))
    # #         # ex_time = fraction_str_to_float(str(shutter_speed))
    # #         # one_stack_shutter_speeds_temp_list.append(round(ex_time,3))
    # if len(one_stack_ims_temp_list) == NUMBER_OF_IMAGES_PER_STACK:
    #     list_of_ims.append(one_stack_ims_temp_list)
    #     #             # list_of_isos.append(one_stack_isos_temp_list)
    #     #             # list_of_shutter_speeds.append(one_stack_shutter_speeds_temp_list)
    #     one_stack_ims_temp_list = []


#             # one_stack_isos_temp_list = []
#             # one_stack_shutter_speeds_temp_list = []

def make_raw_im_show_data_one_im(c1,c2,c3,c4,shape0,shape1,im_path):
    params = {
        'input_stage': 'raw',
        #             # options: 'raw', 'normal', 'white_balance', 'demosaic', 'xyz', 'srgb', 'gamma', 'tone'
        'output_stage': 'tone',
        #             # options: 'normal', 'white_balance', 'demosaic', 'xyz', 'srgb', 'gamma', 'tone'
        'demosaic_type': 'EA'
    }
    raw_bayer_downscaled = np.empty((int(shape0 * (1 / 2)), int(shape1* (1 / 2))))
    #
    # raw_bayer_downscaled_shape = (raw_bayer_downscaled.shape[1] // 2, raw_bayer_downscaled.shape[0] // 2)
    raw_bayer_downscaled[::2, ::2] = c1
    raw_bayer_downscaled[1::2, ::2] = c2
    raw_bayer_downscaled[::2, 1::2] = c3
    raw_bayer_downscaled[1::2, 1::2] = c4

    # c1 = raw_bayer_[::8, ::8]
    # c2 = raw_bayer_[1::8, ::8]
    # c3 = raw_bayer_[::8, 1::8]
    # c4 = raw_bayer_[1::8, 1::8]
    # raw_bayer_downscaled = np.empty((int(raw_bayer_.shape[0] * 0.25), int(raw_bayer_.shape[1] * 0.25)))
    # raw_bayer_downscaled[::2, ::2] = c1
    # raw_bayer_downscaled[1::2, ::2] = c2
    # raw_bayer_downscaled[::2, 1::2] = c3
    # raw_bayer_downscaled[1::2, 1::2] = c4


    metadata = get_metadata(im_path)
    #
    output_image = run_pipeline_v2(raw_bayer_downscaled, params, metadata)
    output_image = np.clip(output_image, 0, 1)

    output_image_shape = (int(shape1 * (1 / 7)), int(shape0 * (1 / 7)))
    output_image = cv2.resize(output_image, output_image_shape)

    # output_image = output_image ** (1 / 2.2)
    output_image = output_image * 255
    output_image = output_image.astype(np.uint8)
    return output_image
    # output_image = cv2.cvtColor(output_image,cv2.COLOR_BGR2RGB)
    # cv2.imwrite(read_path + "out_8/"+str(number)+'.jpg',output_image)


def save_im(raw_bayer,im_path):
    # raw_bayer_ = np.right_shift(raw_bayer, 6)
    raw_bayer_ = raw_bayer[54:4544 - 10, 148:6868]
    c1 = raw_bayer_[::4, ::4]
    c2 = raw_bayer_[1::4, ::4]
    c3 = raw_bayer_[::4, 1::4]
    c4 = raw_bayer_[1::4, 1::4]
    shape0 = raw_bayer_.shape[0]
    shape1 = raw_bayer_.shape[1]
    raw_bayer_downscaled_ = make_npy_data_for_algorithem_one_im(c1,c2,c3,c4,shape0,shape1)
    output_im = make_raw_im_show_data_one_im(c1,c2,c3,c4,shape0,shape1,im_path)
    return raw_bayer_downscaled_,output_im



    # c1 = raw_bayer_[::8, ::8]
    # c2 = raw_bayer_[1::8, ::8]
    # c3 = raw_bayer_[::8, 1::8]
    # c4 = raw_bayer_[1::8, 1::8]
    # raw_bayer_downscaled = np.empty((int(raw_bayer_.shape[0] * 0.25), int(raw_bayer_.shape[1] * 0.25)))
    # raw_bayer_downscaled[::2, ::2] = c1
    # raw_bayer_downscaled[1::2, ::2] = c2
    # raw_bayer_downscaled[::2, 1::2] = c3
    # raw_bayer_downscaled[1::2, 1::2] = c4




start_time = time.time()

read_path = 'D:/dngs/Scene12/'
scene_num = '12'

save_loc = os.path.join(os.path.dirname(__file__), 'Image_Arrays_exposure_separate')
os.makedirs(save_loc, exist_ok=True)
save_loc_show = os.path.join(os.path.dirname(__file__), 'Image_Arrays_from_dng_separate')
os.makedirs(save_loc_show, exist_ok=True)
joinPathChar = "/"
#
listdir_ = os.listdir(read_path)
images = [read_path + f for f in listdir_ if f.endswith(('.dng', '.DNG'))]
#
list_of_ims = []
list_of_ims_show = []
# # list_of_shutter_speeds = []
#
if (len(images) % NUMBER_OF_IMAGES_PER_STACK) != 0:
    print("Wrong number of input images or Wrong number of images per stack!")
else:
    number_of_stacks = int(len(images) / NUMBER_OF_IMAGES_PER_STACK)
    images.sort()
    one_stack_ims_temp_list = []
    show_one_stack_ims_temp_list = []
#     # one_stack_isos_temp_list = []
#     # one_stack_shutter_speeds_temp_list = []
    count = 0
    count = 33*40
    i = 0
    i = 33*15
    j = i+1
    k = 0
    image_path1 = images[i]
    im1,mean1,output_im_algorithm1,output_im_show1 = get_normed_im(image_path1)
    #means = []
    means.append(mean1)
    one_stack_ims_temp_list.append(output_im_algorithm1)
    show_one_stack_ims_temp_list.append(output_im_show1)
    print(count)
    count += 1
    #- 38*40
    while count < TOTAL_IMS  + 1:
        image_path2 = images[j]
        im2, mean2, output_im_algorithm2, output_im_show2 = get_normed_im(image_path2)
        if NEW_SCALES[k] == SCALE_LABELS[i % NUMBER_OF_IMAGES_PER_STACK]:
            k += 1
            # if k == len(NEW_SCALES):
            #     break
        elif NEW_SCALES[k] == SCALE_LABELS[j % NUMBER_OF_IMAGES_PER_STACK]:
            i = j
            im1 = im2
            image_path1 = image_path2
            means.append(mean2)
            one_stack_ims_temp_list.append(output_im_algorithm2)
            show_one_stack_ims_temp_list.append(output_im_show2)
            if len(one_stack_ims_temp_list) == NUMBER_OF_IMAGES_PER_STACK_NEW:
                # list_of_ims.append(one_stack_ims_temp_list)
                # list_of_ims_show.append(show_one_stack_ims_temp_list)
                frame_number = count // NUMBER_OF_IMAGES_PER_STACK_NEW
                frame_number_s = str(frame_number)
                if len(frame_number_s) == 1:
                    frame_number_s = '0' + frame_number_s
                np.save(save_loc + joinPathChar + 'Scene' + scene_num + '_ds_raw_imgs' + frame_number_s, np.asarray(one_stack_ims_temp_list))
                np.save(save_loc_show + joinPathChar + 'Scene' + scene_num +'_show_dng_imgs'+ frame_number_s,
                        np.asarray(show_one_stack_ims_temp_list))
                one_stack_ims_temp_list=[]
                show_one_stack_ims_temp_list=[]
                k = 0
            else:
                k += 1
            print(count)
            count += 1
            j += 1
            # if j == len(SCALE_LABELS):
            #     break
            #k += 1

        elif SCALE_LABELS[i % NUMBER_OF_IMAGES_PER_STACK] > NEW_SCALES[k] > SCALE_LABELS[j % NUMBER_OF_IMAGES_PER_STACK]:
            print("i: "+ str(i)+" j: "+str(j)+" k: "+str(k))
            #im_path = 'D:\Final_dng\Scene19_Blackspace_dng/1P0A0038.dng'
            ind1 = i % NUMBER_OF_IMAGES_PER_STACK
            ind2 = j % NUMBER_OF_IMAGES_PER_STACK
            targetx = NEW_SCALES[k]
            normed_im = method(im1,im2,ind1,ind2,targetx)
            means.append(np.mean(normed_im))
            print(count)
            count += 1
            im = normed_im * (2 ** 14 - 1)
            image_path = image_path1 if (SCALE_LABELS[i % NUMBER_OF_IMAGES_PER_STACK] - NEW_SCALES[k]) <= (NEW_SCALES[k] - SCALE_LABELS[j % NUMBER_OF_IMAGES_PER_STACK]) else image_path2
            output_im_algorithm,output_im_show=save_im(im, image_path)
            one_stack_ims_temp_list.append(output_im_algorithm)
            show_one_stack_ims_temp_list.append(output_im_show)

            k += 1
    print(means)
    #np.save(save_loc + joinPathChar + 'Scene' + scene_num + '_ds_raw_imgs', np.asarray(list_of_ims))
    #np.save(save_loc_show + joinPathChar + 'Scene' + scene_num + '_show_dng_imgs', np.asarray(list_of_ims_show))
print("running time: {:.3f}".format(time.time()-start_time))
