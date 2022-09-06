import cv2
import numpy as np
import rawpy
from simple_camera_pipeline.python.pipeline import run_pipeline_v2,get_metadata

SCALE_LABELS = [15,8,6,4,2,1,1/2,1/4,1/8,1/15,1/30,1/60,1/125,1/250,1/500]

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
    raw_bayer = raw_bayer/(2**14-1)
    np.clip(raw_bayer, 0, 1, out=raw_bayer)
    return raw_bayer


def method_1_assume_line_across_origin(image_path1,image_path2,ind1,ind2,targetx):
    im1 = get_normed_im(image_path1)
    im2 = get_normed_im(image_path2)
    x1 = SCALE_LABELS[ind1]
    x2 = SCALE_LABELS[ind2]
    return one_pixel_function(im1,im2,x1,x2,targetx)

def method_1_assume_line_across_origin_2(image_path1,image_path2,ind1,ind2,targetx):
    im1 = get_normed_im(image_path1)
    im2 = get_normed_im(image_path2)
    x1 = SCALE_LABELS[ind1]
    x2 = SCALE_LABELS[ind2]
    return one_pixel_function_2(im1,im2,x1,x2,targetx)

def compare_approxi_caputured(im_path,normed_im):
    captured = get_normed_im(im_path)
    diff = np.abs(captured-normed_im)
    m = np.max(diff)
    min = np.min(diff)
    diff_sum = np.sum(diff)
    captured = captured*(2**14-1)
    show_im(captured,im_path)
    return diff_sum

def show_im(raw_bayer,im_path):
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
    cv2.imwrite("capthured_s19_1P0A0038.jpg",output_image)


x = np.array([1,1,0.8,0.4,1,0.9])
y = np.array([1,0.3,0.2,0.06,0.9,0.2])

z = one_pixel_function(x,y,4,1,2)
print(z)

image_path1 = 'D:\Final_dng\Scene19_Blackspace_dng/1P0A0037.dng'
image_path2 = 'D:\Final_dng\Scene19_Blackspace_dng/1P0A0039.dng'
im_path = 'D:\Final_dng\Scene19_Blackspace_dng/1P0A0038.dng'
ind1 = 6
ind2 = 8
targetx = 1/4

#normed_im = method_1_assume_line_across_origin(image_path1,image_path2,ind1,ind2,targetx)
# diff_sum = compare_approxi_caputured(im_path,normed_im)

normed_im2 = method_1_assume_line_across_origin_2(image_path1,image_path2,ind1,ind2,targetx)
diff_sum2 = compare_approxi_caputured(im_path,normed_im2)
im2 = normed_im2*(2**14-1)
#show_im(im2,im_path)
print(diff_sum2)