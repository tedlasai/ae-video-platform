import os
import time

import cv2
import numpy as np
import rawpy
from simple_camera_pipeline.python.pipeline import run_pipeline_v2,get_metadata

SCALE_LABELS = [15,8,6,4,2,1,1/2,1/4,1/8,1/15,1/30,1/60,1/125,1/250,1/500]
NEW_SCALES = [15,13,10,8,6,5,4,3.2,2.5,2,1.6,1.3,1,0.8,0.6,0.5,0.4,0.3,1/4,1/5,1/6,1/8,1/10,1/13,1/15,1/20,1/25,1/30,1/40,1/50,1/60,1/80,1/100,1/125,1/160,1/200,1/250,1/320,1/400,1/500]
means = []
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
    save_im(raw_bayer, image_path,len(means))
    raw_bayer = raw_bayer/(2**14-1)
    np.clip(raw_bayer, 0, 1, out=raw_bayer)
    mean = np.mean(raw_bayer)
    return raw_bayer,mean


def method_1_assume_line_across_origin(image_path1,image_path2,ind1,ind2,targetx):
    im1,mean1 = get_normed_im(image_path1)
    im2,mean2 = get_normed_im(image_path2)
    x1 = SCALE_LABELS[ind1]
    x2 = SCALE_LABELS[ind2]
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

def compare_approxi_caputured(im_path,normed_im):
    captured,mean = get_normed_im(im_path)
    diff = np.abs(captured-normed_im)
    m = np.max(diff)
    min = np.min(diff)
    diff_sum = np.sum(diff)
    captured = captured*(2**14-1)
    save_im(captured,im_path)
    return diff_sum

def save_im(raw_bayer,im_path,number):
    params = {
        'input_stage': 'raw',
        #             # options: 'raw', 'normal', 'white_balance', 'demosaic', 'xyz', 'srgb', 'gamma', 'tone'
        'output_stage': 'tone',
        #             # options: 'normal', 'white_balance', 'demosaic', 'xyz', 'srgb', 'gamma', 'tone'
        'demosaic_type': 'EA'
    }

    # raw_bayer_ = np.right_shift(raw_bayer, 6)
    raw_bayer_ = raw_bayer[54:4544 - 10, 148:6868]
    c1 = raw_bayer_[::4, ::4]
    c2 = raw_bayer_[1::4, ::4]
    c3 = raw_bayer_[::4, 1::4]
    c4 = raw_bayer_[1::4, 1::4]
    raw_bayer_downscaled = np.empty((int(raw_bayer_.shape[0] * (1 / 2)), int(raw_bayer_.shape[1] * (1 / 2))))

    raw_bayer_downscaled_shape = (raw_bayer_downscaled.shape[1] // 2, raw_bayer_downscaled.shape[0] // 2)
    raw_bayer_downscaled[::2, ::2] = c1
    raw_bayer_downscaled[1::2, ::2] = c2
    raw_bayer_downscaled[::2, 1::2] = c3
    raw_bayer_downscaled[1::2, 1::2] = c4

    metadata = get_metadata(im_path)
    #
    output_image = run_pipeline_v2(raw_bayer_downscaled, params, metadata)
    output_image = np.clip(output_image, 0, 1)

    output_image_shape = (int(raw_bayer_.shape[1] * (1 / 7)), int(raw_bayer_.shape[0] * (1 / 7)))
    output_image = cv2.resize(output_image, output_image_shape)

    # output_image = output_image ** (1 / 2.2)
    output_image = output_image * 255
    output_image = output_image.astype(np.uint8)
    output_image = cv2.cvtColor(output_image,cv2.COLOR_BGR2RGB)
    cv2.imwrite(read_path + "out_3/"+str(number)+'.jpg',output_image)


# x = np.array([1,1,0.8,0.4,1,0.9])
# y = np.array([1,0.3,0.2,0.06,0.9,0.2])
#
# z = one_pixel_function(x,y,4,1,2)
# print(z)


#show_im(im2,im_path)



start_time = time.time()
NUMBER_OF_IMAGES_PER_STACK = 15
read_path = 'D:/project_data/4d_exposure/Final_dng/Scene1_Frame1/'
scene_num = '1'
# save_loc = os.path.join(os.path.dirname(__file__), 'Image_Arrays_from_dng')
# os.makedirs(save_loc, exist_ok=True)
# joinPathChar = "/"
#
listdir_ = os.listdir(read_path)
images = [read_path + f for f in listdir_ if f.endswith(('.dng', '.DNG'))]
#
list_of_ims = []
# # list_of_isos = []
# # list_of_shutter_speeds = []
#
if (len(images) % NUMBER_OF_IMAGES_PER_STACK) != 0:
    print("Wrong number of input images or Wrong number of images per stack!")
else:
    number_of_stacks = int(len(images) / NUMBER_OF_IMAGES_PER_STACK)
    images.sort()
    #one_stack_ims_temp_list = []
#     # one_stack_isos_temp_list = []
#     # one_stack_shutter_speeds_temp_list = []
    count = 0
    i = 0
    j = 1
    k = 0
    image_path1 = images[i]
    im1,mean1 = get_normed_im(image_path1)
    #means = []
    means.append(mean1)
    print(count)
    count += 1
    while j < NUMBER_OF_IMAGES_PER_STACK:
        image_path2 = images[j]
        im2, mean2 = get_normed_im(image_path2)
        if NEW_SCALES[k] == SCALE_LABELS[i]:
            k += 1
            if k == len(NEW_SCALES):
                break
        elif NEW_SCALES[k] == SCALE_LABELS[j]:
            i = j
            im1 = im2
            image_path1 = image_path2
            means.append(mean2)
            print(count)
            count += 1
            j += 1
            if j == len(SCALE_LABELS):
                break
            k += 1

        elif SCALE_LABELS[i] > NEW_SCALES[k] > SCALE_LABELS[j]:
            print("i: "+ str(i)+" j: "+str(j)+" k: "+str(k))
            #im_path = 'D:\Final_dng\Scene19_Blackspace_dng/1P0A0038.dng'
            ind1 = i
            ind2 = j
            targetx = NEW_SCALES[k]
            normed_im = method(im1,im2,ind1,ind2,targetx)
            means.append(np.mean(normed_im))
            print(count)
            count += 1
            im = normed_im * (2 ** 14 - 1)
            image_path = image_path1 if (SCALE_LABELS[i] - NEW_SCALES[k]) <= (NEW_SCALES[k] - SCALE_LABELS[j]) else image_path2
            save_im(im, image_path, len(means)-1)
            print(means)
            k += 1

print("running time: {:.3f}".format(time.time()-start_time))
