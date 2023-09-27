import cv2
import numpy as np

from exposure_general import Exposure
from skimage.measure import shannon_entropy

from update_visulization import get_hists


class ExposureEntropy(Exposure):
    def __init__(self,
                 raw_images,
                 srgb_imgs,

                 num_hist_bins=100,

                 start_index=20, ):
        # self.gender = gender
        # Prototype initialization 3.x:
        self.r_percent = 0.2126,
        self.g_percent = 0.7152,
        self.b_percent = 0.0722,
        self.srgb_imgs = srgb_imgs
        super().__init__(
            raw_images,
            srgb_imgs,

            num_hist_bins=num_hist_bins,

            start_index=start_index, )

    def imput_imgs_processing(self):
        # raw_bayer = np.load(self.srgb_images)
        # raw_bayer = raw_bayer[:, :, ::8, ::8, :]  # downsize 1/8 * 1/8 = 1/64, to be modified based on downsample size!!
        current_rgb_img = self.srgb_imgs / (self.absolute_bit - 1)
        current_rgb_img[:, :, :, :, 0] = current_rgb_img[:, :, :, :, 0] * self.r_percent  # 0.2126 by default
        current_rgb_img[:, :, :, :, 1] = current_rgb_img[:, :, :, :, 1] * self.g_percent  # 0.7152 by default
        current_rgb_img[:, :, :, :, 2] = current_rgb_img[:, :, :, :, 2] * self.b_percent  # 0.0722 by default
        rgb_blended_ims = np.sum(current_rgb_img, axis=4)
        return rgb_blended_ims

    def pipeline(self):
        rgb_blended_ims = self.imput_imgs_processing()
        _, _, height, width = rgb_blended_ims.shape
        downsampled_ims1 = np.reshape(rgb_blended_ims, (self.num_frame, self.num_ims_per_frame, height * width))
        opti_inds = []
        ind = self.start_index
        opti_inds.append(ind)

        for j in range(1, self.num_frame):
            current_frame = downsampled_ims1[j]
            entropies = np.empty(self.num_ims_per_frame)
            for i in range(self.num_ims_per_frame):
                current_frame_exposure = current_frame[i]
                # the default output_image_shape is (168, 112)
                # output_image_shape = (self.w, self.h)
                #
                # current_frame_exposure = cv2.resize(current_frame_exposure, output_image_shape)
                current_frame_exposure = current_frame_exposure.flatten()
                # raw_frame_exposure = raw_frame[i].flatten()
                thresholded_current_frame = current_frame_exposure  # [raw_frame_exposure<self.high_threshold]

                # thresholded_current_frame = current_frame_exposure[~np.isnan(current_frame)]

                entropies[i] = shannon_entropy(thresholded_current_frame)  # *(1/self.high_threshold)

            ind = np.argmax(entropies)
            opti_inds.append(ind)

        opti_inds_adjusted_previous_n_frames = self.adjusted_opti_inds_v2_by_average_of_previous_n_frames(opti_inds)

        flatten_ims = np.reshape(self.raw_imgs, (self.num_frame, self.num_ims_per_frame, self.h * self.w))
        hists, _ = get_hists(flatten_ims)
        weighted_means = np.mean(flatten_ims,axis=2)
        print(np.round(opti_inds_adjusted_previous_n_frames).astype(int))

        return opti_inds_adjusted_previous_n_frames, opti_inds, weighted_means, hists
