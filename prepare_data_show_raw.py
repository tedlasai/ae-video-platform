import os
import time

import rawpy
import numpy as np
import cv2
from simple_camera_pipeline_old.python.pipeline import run_pipeline_v2,get_metadata

# im_path = "C:/Users/beixuan/Documents/testAdobedng/1P0A5070.dng"
# im_path2 = "C:/Users/beixuan/Documents/testAdobedng/1P0A5070.cr2"
# raw_im = rawpy.imread(im_path)
# #print("raw_im shape: "+str(raw_im.shape))
# raw_bayer = raw_im.raw_image.copy()
# print("raw_bayer shape: "+str(raw_bayer.shape))
# print("max1: "+str(np.max(raw_bayer)))
# print("min1: "+str(np.min(raw_bayer)))
# raw_im2 = rawpy.imread(im_path2)
# #print("raw_im shape: "+str(raw_im.shape))
# raw_bayer2 = raw_im2.raw_image.copy()
# print("raw_bayer2 shape: "+str(raw_bayer2.shape))
# #scale data between 0-255
# #this is the active area for this sensor and also takes into account the default crop
# raw_bayer = raw_bayer [54:4544-10,148:6868]
# #print(np.right_shift(np.array([16383], 6)))
# print(raw_bayer.shape)
# #raw_bayer_downscaled = np.right_shift(raw_bayer, 6)
# c1 = raw_bayer[::16,::16]
# c2 = raw_bayer[1::16,::16]
# c3 = raw_bayer[::16,1::16]
# c4 = raw_bayer[1::16,1::16]
# raw_bayer_downscaled = np.empty((int(raw_bayer.shape[0]*1/8),int(raw_bayer.shape[1]*1/8)))
# raw_bayer_downscaled[::2,::2] = c1
# raw_bayer_downscaled[1::2,::2] = c2
# raw_bayer_downscaled[::2,1::2] = c3
# raw_bayer_downscaled[1::2,1::2] = c4
#
# print(raw_bayer_downscaled.shape)
# #print(raw_bayer_downscaled)
#
# print("max2: "+str(np.max(raw_bayer2)))
# print("min2: "+str(np.min(raw_bayer2)))
# raw_bayer2 = raw_bayer2[54:4544-10,148:6868]
# #print(np.right_shift(np.array([16383], 6)))
# print(raw_bayer2.shape)
# #raw_bayer_downscaled = np.right_shift(raw_bayer, 6)
# c12 = raw_bayer2[::16,::16]
# c22 = raw_bayer2[1::16,::16]
# c32 = raw_bayer2[::16,1::16]
# c42 = raw_bayer2[1::16,1::16]
# raw_bayer_downscaled2 = np.empty((int(raw_bayer2.shape[0]*1/8),int(raw_bayer2.shape[1]*1/8)))
# raw_bayer_downscaled2[::2,::2] = c12
# raw_bayer_downscaled2[1::2,::2] = c22
# raw_bayer_downscaled2[::2,1::2] = c32
# raw_bayer_downscaled2[1::2,1::2] = c42
#
# print(raw_bayer_downscaled2.shape)
# #print(raw_bayer_downscaled2)
#
# print(np.array_equal(raw_bayer_downscaled2,raw_bayer_downscaled))
# #raw_bayer_downscaled = raw_bayer
# params = {
#     'input_stage': 'raw',  # options: 'raw', 'normal', 'white_balance', 'demosaic', 'xyz', 'srgb', 'gamma', 'tone'
#     'output_stage': 'demosaic',  # options: 'normal', 'white_balance', 'demosaic', 'xyz', 'srgb', 'gamma', 'tone'
#     'demosaic_type': 'menon2007',
#
# }
# metadata = get_metadata(im_path)
#
# output_image = run_pipeline_v2(raw_bayer_downscaled, params,metadata)
# output_image = np.clip(output_image,0,1)
# output_image = output_image**(1/2.2)
# output_image = output_image*255
# output_image = output_image.astype(np.uint8)
# output_image = cv2.cvtColor(output_image,cv2.COLOR_BGR2RGB)
# cv2.imwrite("out.jpg",output_image)



start_time = time.time()
NUMBER_OF_IMAGES_PER_STACK = 15
read_path = 'D:/project_data/4d_exposure/Final_dng/Scene4_BackAndRightLight_dng/'
scene_num = '4'
save_loc = os.path.join(os.path.dirname(__file__), 'Image_Arrays_from_dng')
os.makedirs(save_loc, exist_ok=True)
joinPathChar = "/"
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
    one_stack_ims_temp_list = []
#     # one_stack_isos_temp_list = []
#     # one_stack_shutter_speeds_temp_list = []
    count = 0
    for im_path in images:
        #im_path = 'D:/Final_dng/Scene8_Window_dng/1P0X0005.dng'
        count += 1
        print(count)
        raw_im = rawpy.imread(im_path)
        raw_bayer = raw_im.raw_image.copy()
#
#
#
#         #scale data between 0-255
#         #this is the active area for this sensor and also takes into account the default crop
        raw_bayer_ = raw_bayer[54:4544-10,148:6868]
#         #print(np.right_shift(np.array([16383], 6)))
#
        #raw_bayer_ = np.right_shift(raw_bayer, 6)
#         c1 = raw_bayer_[::14, ::14]
#         c2 = raw_bayer_[1::14, ::14]
#         c3 = raw_bayer_[::14, 1::14]
#         c4 = raw_bayer_[1::14, 1::14]
#         raw_bayer_downscaled = np.empty((int(raw_bayer_.shape[0] * (1/7)), int(raw_bayer_.shape[1] * (1/7))))
#
#         # c1 = raw_bayer_[::16, ::16]
#         # c2 = raw_bayer_[1::16, ::16]
#         # c3 = raw_bayer_[::16, 1::16]
#         # c4 = raw_bayer_[1::16, 1::16]
#         # raw_bayer_downscaled = np.empty((int(raw_bayer_.shape[0] * (1/8)), int(raw_bayer_.shape[1] * (1/8))))
#         raw_bayer_downscaled[::2, ::2] = c1
#         raw_bayer_downscaled[1::2, ::2] = c2
#         raw_bayer_downscaled[::2, 1::2] = c3
#         raw_bayer_downscaled[1::2, 1::2] = c4
#
#
# #
# #         # raw_bayer_downscaled = raw_bayer
#         params = {
#             'input_stage': 'raw',
# #             # options: 'raw', 'normal', 'white_balance', 'demosaic', 'xyz', 'srgb', 'gamma', 'tone'
#             'output_stage': 'tone',
# #             # options: 'normal', 'white_balance', 'demosaic', 'xyz', 'srgb', 'gamma', 'tone'
#             'demosaic_type': 'menon2007'
#         }
#         metadata = get_metadata(im_path)
# #
#         output_image = run_pipeline_v2(raw_bayer_downscaled, params, metadata)
#         output_image = np.clip(output_image, 0, 1)
#        # output_image = output_image ** (1 / 2.2)
#         output_image = output_image * 255
#         output_image = output_image.astype(np.uint8)
        #output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
        # output_image = cv2.cvtColor(output_image,cv2.COLOR_BGR2RGB)
        #cv2.imwrite("outgamma04516.jpg",output_image)
        #break

        params = {
            'input_stage': 'raw',
            #             # options: 'raw', 'normal', 'white_balance', 'demosaic', 'xyz', 'srgb', 'gamma', 'tone'
            'output_stage': 'tone',
            #             # options: 'normal', 'white_balance', 'demosaic', 'xyz', 'srgb', 'gamma', 'tone'
            'demosaic_type': 'EA'
        }

        # raw_bayer_ = np.right_shift(raw_bayer, 6)
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

#
#         raw_bayer_downscaled = raw_bayer_downscaled.astype(np.uint8)
        one_stack_ims_temp_list.append(output_image)
#         # shutter_speed,iso = getMata_exifread(im_path)
#         # one_stack_isos_temp_list.append(int(str(iso)))
#         # ex_time = fraction_str_to_float(str(shutter_speed))
#         # one_stack_shutter_speeds_temp_list.append(round(ex_time,3))
        if len(one_stack_ims_temp_list) == NUMBER_OF_IMAGES_PER_STACK:
            list_of_ims.append(one_stack_ims_temp_list)
            # list_of_isos.append(one_stack_isos_temp_list)
#             # list_of_shutter_speeds.append(one_stack_shutter_speeds_temp_list)
            one_stack_ims_temp_list = []
#             # one_stack_isos_temp_list = []
#             # one_stack_shutter_speeds_temp_list = []
#
    np.save(save_loc + joinPathChar + 'Scene' + scene_num + '_show_dng_imgs',np.asarray(list_of_ims))
#     # np.save('Scene_img_ex_times',np.asarray(list_of_shutter_speeds))
#     # np.save('Scene_img_isos',np.asarray(list_of_isos))
print("running time: {:.3f}".format(time.time()-start_time))