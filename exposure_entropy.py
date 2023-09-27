from exposure_general import Exposure
from skimage.measure import shannon_entropy
from update_visulization import get_hists

import numpy as np


class ExposureEntropy(Exposure):
    def __init__(self,
                 raw_images,
                 srgb_imgs,
                 start_index=20):
        self.r_percent = 0.2126,
        self.g_percent = 0.7152,
        self.b_percent = 0.0722,
        self.srgb_imgs = srgb_imgs
        super().__init__(
            raw_images,
            start_index=start_index)

    def input_imgs_processing(self):
        current_rgb_img = self.srgb_imgs / (self.absolute_bit - 1)
        current_rgb_img[:, :, :, :, 0] = current_rgb_img[:, :, :, :, 0] * self.r_percent  # 0.2126 by default
        current_rgb_img[:, :, :, :, 1] = current_rgb_img[:, :, :, :, 1] * self.g_percent  # 0.7152 by default
        current_rgb_img[:, :, :, :, 2] = current_rgb_img[:, :, :, :, 2] * self.b_percent  # 0.0722 by default
        rgb_blended_ims = np.sum(current_rgb_img, axis=4)
        return rgb_blended_ims

    def pipeline(self):
        rgb_blended_ims = self.input_imgs_processing()
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
                current_frame_exposure = current_frame_exposure.flatten()
                threshold_current_frame = current_frame_exposure
                entropies[i] = shannon_entropy(threshold_current_frame)
            ind = np.argmax(entropies)
            opti_inds.append(ind)
        opti_inds_adjusted_previous_n_frames = self.adjusted_opti_inds_v2_by_average_of_previous_n_frames(opti_inds)
        flatten_ims = np.reshape(self.raw_imgs, (self.num_frame, self.num_ims_per_frame, self.h * self.w))
        hists, _ = get_hists(flatten_ims)
        weighted_means = np.mean(flatten_ims,axis=2)
        opti_inds_adjusted_previous_n_frames = (np.round(opti_inds_adjusted_previous_n_frames).astype(int))
        return opti_inds_adjusted_previous_n_frames, opti_inds, weighted_means, hists
